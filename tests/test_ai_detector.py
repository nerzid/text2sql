from unittest.mock import patch, MagicMock

from app.services.ai_detector import is_ai_generated


def test_is_ai_generated():
    with patch("app.services.ai_detector._pipe") as mock_pipe:
        # Setup mock
        mock_model = MagicMock()
        mock_pipe.model = mock_model
        mock_pipe.model.config.id2label = {0: "HUMAN", 1: "AI"}

        # Test AI generated text
        mock_outputs = MagicMock()
        mock_outputs.logits = MagicMock()
        mock_model.return_value = mock_outputs

        # Mock the softmax function to return AI prediction
        with patch("torch.nn.functional.softmax") as mock_softmax:
            mock_probs = MagicMock()
            mock_probs[1].item.return_value = 0.8
            mock_softmax.return_value = [mock_probs]

            with patch("torch.argmax") as mock_argmax:
                mock_argmax.return_value.item.return_value = 1

                result = is_ai_generated("This is a test text")
                assert result is True
