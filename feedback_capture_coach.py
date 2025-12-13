import cv2

class HumanFeedbackCaptureCoach:
    """
    Captures human feedback for DeepCOACH algorithm
    Uses keyboard input to provide evaluative feedback on agent behavior
    """
    def __init__(self):
        self.current_feedback = None
        self.feedback_history = []
        self.last_feedback_step = -999  

    def reset(self):
        """Reset feedback tracking for new episode"""
        self.last_feedback_step = -999
        
    def capture_feedback(self, current_step, min_interval=15):
        """
        Capture keyboard feedback for DeepCOACH
        
        Args:
            current_step: current timestep
            min_interval: minimum steps between feedbacks to avoid spam
        
        Returns:
            feedback: +1.0 (good), -1.0 (bad), 0.0 (neutral), or None if no feedback
        """
        key = cv2.waitKey(1) & 0xFF
        
        # Enforce minimum interval between feedbacks
        if current_step - self.last_feedback_step < min_interval:
            return None
        
        feedback = None
        
        if key == ord('g') or key == ord('G'):  # Good action
            feedback = +1.0
            print(f"[Step {current_step}] Feedback: +1.0 (GOOD)")
            
        elif key == ord('b') or key == ord('B'):  # Bad action
            feedback = -1.0
            print(f"[Step {current_step}] Feedback: -1.0 (BAD)")
            
        elif key == ord('n') or key == ord('N'):  # Neutral (won't store window)
            feedback = 0.0
            print(f"[Step {current_step}] Feedback: 0.0 (NEUTRAL - no window stored)")
        
        # Update state if feedback was given
        if feedback is not None:
            self.last_feedback_step = current_step
            self.feedback_history.append({
                'step': current_step,
                'feedback': feedback
            })
        
        return feedback
    
    def get_stats(self):
        """Get statistics about feedback provided"""
        if not self.feedback_history:
            return {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        positive = sum(1 for f in self.feedback_history if f['feedback'] == 1.0)
        negative = sum(1 for f in self.feedback_history if f['feedback'] == -1.0)
        neutral = sum(1 for f in self.feedback_history if f['feedback'] == 0.0)
        
        return {
            'total': len(self.feedback_history),
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'history': self.feedback_history
        }
