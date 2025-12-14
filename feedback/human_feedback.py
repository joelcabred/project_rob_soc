

class HumanFeedbackCapture:
    def __init__(self, backend, min_interval=15):
        self.backend = backend
        self.min_interval = min_interval

        self.last_feedback_step = -999
        self.feedback_history = []
        self.current_feedback = None
        print('USING KEYBOARD FEEDBACK')
        print('+ : Positive')
        print('0 : Neutral')
        print('- : Negative')


    def reset(self):
        self.last_feedback_step = -999
        self.backend.reset()

    def capture_feedback(self, step):
        if step - self.last_feedback_step < self.min_interval:
            return None

        fb = self.backend.poll()
        if fb is None:
            return None

        self.last_feedback_step = step
        self.current_feedback = fb
        self.feedback_history.append({
            "step": step,
            "feedback": fb
        })

        return fb

    def get_stats(self):
        return {
            "total": len(self.feedback_history),
            "positive": sum(f["feedback"] == 1 for f in self.feedback_history),
            "negative": sum(f["feedback"] == -1 for f in self.feedback_history),
            "neutral":  sum(f["feedback"] == 0 for f in self.feedback_history),
        }

    def close(self):
        self.backend.close()
