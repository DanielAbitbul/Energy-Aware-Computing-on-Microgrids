
from workload.visualization import (
    plot_training_history,
    plot_predictions,
    plot_residuals,
    plot_uncertainty,
    create_all_plots,
)


def sample_data():
    history = {
        "train_loss": [1.2, 0.9, 0.7],
        "validation_loss": [1.3, 1.0, 0.8],
    }

    actual = [10.0, 15.0, 20.0, 25.0]
    predicted_mean = [11.0, 14.0, 19.0, 26.0]
    predicted_std = [2.0, 2.5, 3.0, 2.0]

    return history, actual, predicted_mean, predicted_std


def test_plot_training_history_creates_file(tmp_path):
    history, _, _, _ = sample_data()
    output_path = tmp_path / "training_loss.png"

    plot_training_history(history, output_path)

    assert output_path.exists()


def test_plot_predictions_creates_file(tmp_path):
    _, actual, mean, _ = sample_data()
    output_path = tmp_path / "predictions.png"

    plot_predictions(actual, mean, output_path)

    assert output_path.exists()


def test_plot_residuals_creates_file(tmp_path):
    _, actual, mean, _ = sample_data()
    output_path = tmp_path / "residuals.png"

    plot_residuals(actual, mean, output_path)

    assert output_path.exists()


def test_plot_uncertainty_creates_file(tmp_path):
    _, actual, mean, std = sample_data()
    output_path = tmp_path / "uncertainty.png"

    plot_uncertainty(actual, mean, std, output_path)

    assert output_path.exists()


def test_create_all_plots_creates_all_files(tmp_path):
    history, actual, mean, std = sample_data()

    create_all_plots(
        history,
        actual,
        mean,
        std,
        tmp_path,
    )

    expected_files = [
        "training_loss.png",
        "predictions.png",
        "residuals.png",
        "uncertainty.png",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
