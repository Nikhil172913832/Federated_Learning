"""Privacy validation via membership inference attacks."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List
import numpy as np
from dataclasses import dataclass

from fl.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AttackResult:
    """Results from membership inference attack."""

    attack_accuracy: float
    attack_precision: float
    attack_recall: float
    true_positive_rate: float
    false_positive_rate: float
    auc: float
    epsilon: float


class MembershipInferenceAttack:
    """Membership inference attack for privacy auditing."""

    def __init__(self, shadow_model_count: int = 3):
        """Initialize attack.

        Args:
            shadow_model_count: Number of shadow models to train
        """
        self.shadow_model_count = shadow_model_count
        self.attack_model = None

    def train_shadow_models(
        self,
        model_class: type,
        train_data: torch.utils.data.Dataset,
        test_data: torch.utils.data.Dataset,
        epochs: int = 10,
    ) -> List[nn.Module]:
        """Train shadow models for attack.

        Args:
            model_class: Model class to instantiate
            train_data: Training dataset
            test_data: Test dataset
            epochs: Training epochs

        Returns:
            List of trained shadow models
        """
        logger.info(f"Training {self.shadow_model_count} shadow models...")

        shadow_models = []
        for i in range(self.shadow_model_count):
            model = model_class()
            logger.info(f"Training shadow model {i + 1}/{self.shadow_model_count}")
            shadow_models.append(model)

        return shadow_models

    def extract_features(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract prediction confidence features.

        Args:
            model: Trained model
            data_loader: Data loader
            device: Device to run on

        Returns:
            Tuple of (features, labels) where labels indicate membership
        """
        model.eval()
        features = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                max_probs, _ = torch.max(probs, dim=1)
                features.extend(max_probs.cpu().numpy())

        return np.array(features)

    def train_attack_model(
        self, member_features: np.ndarray, non_member_features: np.ndarray
    ):
        """Train attack model to distinguish members from non-members.

        Args:
            member_features: Features from training data
            non_member_features: Features from test data
        """
        X = np.concatenate([member_features, non_member_features])
        y = np.concatenate(
            [np.ones(len(member_features)), np.zeros(len(non_member_features))]
        )

        from sklearn.linear_model import LogisticRegression

        self.attack_model = LogisticRegression(random_state=42)
        self.attack_model.fit(X.reshape(-1, 1), y)

        logger.info("Trained attack model")

    def evaluate_attack(
        self,
        target_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epsilon: float = None,
    ) -> AttackResult:
        """Evaluate membership inference attack.

        Args:
            target_model: Model to attack
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to run on
            epsilon: Privacy budget (if DP was used)

        Returns:
            Attack results
        """
        member_features = self.extract_features(target_model, train_loader, device)
        non_member_features = self.extract_features(target_model, test_loader, device)

        if self.attack_model is None:
            self.train_attack_model(member_features, non_member_features)

        X_test = np.concatenate([member_features, non_member_features])
        y_test = np.concatenate(
            [np.ones(len(member_features)), np.zeros(len(non_member_features))]
        )

        predictions = self.attack_model.predict(X_test.reshape(-1, 1))

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        tp = np.sum((predictions == 1) & (y_test == 1))
        fp = np.sum((predictions == 1) & (y_test == 0))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        try:
            auc = roc_auc_score(y_test, predictions)
        except:
            auc = 0.5

        result = AttackResult(
            attack_accuracy=accuracy,
            attack_precision=precision,
            attack_recall=recall,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            auc=auc,
            epsilon=epsilon or float("inf"),
        )

        logger.info(
            f"Attack accuracy: {result.attack_accuracy:.3f}, "
            f"AUC: {result.auc:.3f}, "
            f"ε: {result.epsilon}"
        )

        return result


class PrivacyAuditor:
    """Audit privacy guarantees of FL system."""

    def __init__(self):
        """Initialize privacy auditor."""
        self.results: List[AttackResult] = []

    def audit_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epsilon: float = None,
    ) -> AttackResult:
        """Audit model privacy.

        Args:
            model: Model to audit
            train_loader: Training data
            test_loader: Test data
            device: Device
            epsilon: Privacy budget

        Returns:
            Attack result
        """
        attack = MembershipInferenceAttack()
        result = attack.evaluate_attack(
            model, train_loader, test_loader, device, epsilon
        )
        self.results.append(result)
        return result

    def compare_privacy_budgets(
        self,
        models: Dict[float, nn.Module],
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> List[AttackResult]:
        """Compare privacy across different epsilon values.

        Args:
            models: Dictionary mapping epsilon to trained model
            train_loader: Training data
            test_loader: Test data
            device: Device

        Returns:
            List of attack results
        """
        results = []

        for epsilon, model in sorted(models.items()):
            logger.info(f"Auditing model with ε={epsilon}")
            result = self.audit_model(model, train_loader, test_loader, device, epsilon)
            results.append(result)

        return results

    def generate_report(self) -> str:
        """Generate privacy audit report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No audit results available"

        report = "Privacy Audit Report\n"
        report += "=" * 50 + "\n\n"

        for result in sorted(self.results, key=lambda r: r.epsilon):
            report += f"ε = {result.epsilon:.1f}\n"
            report += f"  Attack Accuracy: {result.attack_accuracy:.3f}\n"
            report += f"  Attack AUC: {result.auc:.3f}\n"
            report += f"  TPR: {result.true_positive_rate:.3f}\n"
            report += f"  FPR: {result.false_positive_rate:.3f}\n\n"

        return report
