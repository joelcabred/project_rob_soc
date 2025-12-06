import cv2

class HumanFeedbackCapture:
    def __init__(self):
        self.current_feedback = None
        self.feedback_history = []
        self.last_feedback_step = -999  # Para evitar feedback muy seguido

    def reset(self):
        self.last_feedback_step = -999
        
    def capture_feedback(self, current_step, min_interval=15):
        """
        It uses keyboard feedback
        
        Args:
            current_step
            min_interval: to avoid spam
        
        Returns:
            feedback: -1, 0, +1, o None if no feedback
        """

        key = cv2.waitKey(1) & 0xFF
        

        if current_step - self.last_feedback_step < min_interval:
            return None
        
        feedback = None
        
        # Mapeo de teclas a feedback prosódico
        if key == ord('+') or key == ord('='):  # Tecla +
            feedback = +1
            print("Feedback: +1 (POSITIVE)")
            
        elif key == ord('-') or key == ord('_'):  # Tecla -
            feedback = -1
            print("Feedback: -1 (NEGATIVE)")
            
        elif key == ord('0'):  # Tecla 0
            feedback = 0
            print("Feedback: 0 (NEUTRAL)")
        
        # Update state if no feedback
        if feedback is not None:
            self.last_feedback_step = current_step
            self.feedback_history.append({
                'step': current_step,
                'feedback': feedback
            })
        
        return feedback
    
    def get_stats(self):
        if not self.feedback_history:
            return {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        positive = sum(1 for f in self.feedback_history if f['feedback'] == 1)
        negative = sum(1 for f in self.feedback_history if f['feedback'] == -1)
        neutral = sum(1 for f in self.feedback_history if f['feedback'] == 0)
        
        return {
            'total': len(self.feedback_history),
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }