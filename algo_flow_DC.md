# Algorithm Flow - Pure Deep COACH (Algorithm 1)

## **Algorithm 1: Deep COACH**

### **Initialization** (Lines 91-106 in coach_mimo.py)
```python
# Create COACH module
coach = DeepCOACHModule(
    input_dim, 
    output_dim,
    window_size=15,           # L: window size
    eligibility_decay=0.9,    # λ: eligibility trace decay
    human_delay=0,            # d: human delay
    learning_rate=0.001,      # α: learning rate
    entropy_coef=0.01,        # β: entropy coefficient
    minibatch_size=4          # m: minibatch size
)
```
**From deep_coach.py (lines 109-125):**
```python
self.policy = GaussianPolicyNetwork(state_size, action_size)  # πθ(a|s)
self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
self.replay_buffer = deque(maxlen=buffer_size)  # E: eligibility replay buffer
self.current_window = []  # w: current window
```

---

### **Episode Loop** (Lines 124-193 in coach_mimo.py)

#### **Step 1: Get action from policy πθ(a|s)** (Line 152)
```python
raw_action, action_prob = coach.predict_action(state, deterministic=False)
```
**Note:** This is where the state `[left,right,qpos,qvel,torque]` goes inside Deep COACH
 
**From deep_coach.py (lines 297-318):**
```python
def predict_action(self, state, deterministic=False):
    state_t = torch.FloatTensor(state).unsqueeze(0)
    
    if deterministic:
        mean, _ = self.policy.forward(state_t)
        action = mean
        log_prob = self.policy.log_proba(state_t, action)
        prob = torch.exp(log_prob).item()
    else:
        # Sample from policy: at ~ πθ(·|st)
        action, log_prob = self.policy.sample(state_t)
        prob = torch.exp(log_prob).item()
    
    return action.detach().numpy()[0], prob
```

**GaussianPolicyNetwork.sample() (lines 48-61):**
```python
def sample(self, state):
    action_mean, log_std = self.forward(state)  # μθ(s), log σθ(s)
    std = log_std.exp()
    
    # Sample from Gaussian: at ~ N(μθ(st), σθ(st))
    normal = torch.distributions.Normal(action_mean, std)
    action = normal.rsample()  # Reparameterization trick
    action = torch.tanh(action)  # Squash to [-1, 1]
    
    log_prob = normal.log_prob(action)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    
    return action, log_prob
```

---

#### **Step 2: Execute action in environment** (Lines 153-156)
```python
action = scale_action(raw_action, env)
action = np.clip(action, env.action_space.low, env.action_space.high)
next_state, _, terminated, truncated, _ = env.step(action)
```

---

#### **Step 3: Record observation (st-d, at-d, pt-d)** (Line 177)
```python
coach.observe_step(state, raw_action, action_prob)
```
**From deep_coach.py (lines 138-150):**
```python
def observe_step(self, state, action, action_prob):
    """Store (st-d, at-d, pt-d) in current window"""
    self.current_window.append({
        'state': state.copy(),      # st-d
        'action': action.copy(),    # at-d
        'prob': action_prob,        # pt-d = πθ_old(at-d|st-d)
        'feedback': 0.0
    })
    self.total_samples += 1
```

---

#### **Step 4: Capture human feedback ft** (Lines 170-173)
```python
feedback = feedback_capture.capture_feedback(steps, min_interval=30)

if feedback is not None:
    coach.give_feedback(feedback)
```
**From deep_coach.py (lines 152-185):**
```python
def give_feedback(self, feedback_value):
    """Human gives feedback ft"""
    if len(self.current_window) == 0:
        return
    
    # Apply human delay d: feedback at time t affects state at time t-d
    delay_idx = max(0, len(self.current_window) - 1 - self.human_delay)
    
    # Store feedback at delayed timestep
    if delay_idx < len(self.current_window):
        self.current_window[delay_idx]['feedback'] = feedback_value
    
    # If ft ≠ 0: truncate window to last L entries
    if feedback_value != 0:
        self.feedback_count += 1
        
        # w ← w[-L:]  (keep last L entries)
        window_to_store = self.current_window[-self.window_size:]
        
        # Add (w, ft) to eligibility replay buffer E
        self.replay_buffer.append({
            'window': window_to_store,
            'final_feedback': feedback_value,
            'feedback_id': self.feedback_count
        })
        self.num_windows += 1
        
        # Reset window: w ← ∅
        self.current_window = []
```

---

#### **Step 5: Train from replay buffer** (Lines 179-182)
```python
if coach.has_valid_model():
    eligibility_norm = coach.train_batch()
    if eligibility_norm is not None:
        eligibility_norms.append(eligibility_norm)
```

**From deep_coach.py (lines 187-287):**

```python
def train_batch(self):
    if len(self.replay_buffer) < self.minibatch_size:
        return None
    
    # Sample minibatch W of m windows from E
    minibatch = random.sample(
        self.replay_buffer, 
        min(self.minibatch_size, len(self.replay_buffer))
    )
    
    # Initialize: ēλ ← 0
    total_eligibility = None
    
    # For each window (w, F) ∈ W:
    for window_data in minibatch:
        window = window_data['window']
        F = window_data['final_feedback']  # Final feedback ft
        
        # Initialize eligibility trace: eλ ← 0
        eligibility_trace = None
        
        # For each (st-d, at-d, pt-d) ∈ w:
        for step in window:
            state = torch.FloatTensor(step['state']).unsqueeze(0)
            action = torch.FloatTensor(step['action']).unsqueeze(0)
            old_prob = step['prob']  # pt-d = πθ_old(at-d|st-d)
            
            # Compute log πθ(at-d|st-d) with current policy
            log_prob_current = self.policy.log_proba(state, action)
            
            # Importance sampling ratio: πθ(at-d|st-d) / pt-d
            importance_ratio = torch.exp(log_prob_current - math.log(old_prob + 1e-10))
            
            # Compute gradient: ∇θ log πθ(at-d|st-d)
            self.optimizer.zero_grad()
            log_prob_current.backward(retain_graph=True)
            
            grad_log_prob = []
            for param in self.policy.parameters():
                if param.grad is not None:
                    grad_log_prob.append(param.grad.clone().flatten())
            grad_log_prob = torch.cat(grad_log_prob)
            
            # Update eligibility trace: eλ ← λ·eλ + (πθ/p)·∇θ log πθ(at-d|st-d)
            if eligibility_trace is None:
                eligibility_trace = importance_ratio.item() * grad_log_prob
            else:
                eligibility_trace = (self.eligibility_decay * eligibility_trace + 
                                   importance_ratio.item() * grad_log_prob)
        
        # Accumulate: ēλ ← ēλ + F·eλ
        if eligibility_trace is not None:
            if total_eligibility is None:
                total_eligibility = F * eligibility_trace
            else:
                total_eligibility += F * eligibility_trace
    
    # Average over minibatch: ēλ ← (1/m)·ēλ
    if total_eligibility is not None:
        total_eligibility = total_eligibility / len(minibatch)
        
        # Add entropy regularization: ēλ ← ēλ + β·∇θ H(πθ(·|st))
        if self.current_window:
            recent_state = torch.FloatTensor(self.current_window[-1]['state']).unsqueeze(0)
        else:
            recent_state = torch.FloatTensor(self.replay_buffer[-1]['window'][-1]['state']).unsqueeze(0)
        
        entropy = self.policy.entropy(recent_state)
        self.optimizer.zero_grad()
        (-entropy).backward()  # Negative to maximize entropy
        
        grad_entropy = []
        for param in self.policy.parameters():
            if param.grad is not None:
                grad_entropy.append(param.grad.clone().flatten())
            else:
                grad_entropy.append(torch.zeros_like(param.flatten()))
        grad_entropy = torch.cat(grad_entropy)
        
        # Size check and correction
        if grad_entropy.size(0) != total_eligibility.size(0):
            if grad_entropy.size(0) < total_eligibility.size(0):
                grad_entropy = torch.cat([grad_entropy, 
                    torch.zeros(total_eligibility.size(0) - grad_entropy.size(0))])
            else:
                grad_entropy = grad_entropy[:total_eligibility.size(0)]
        
        total_eligibility += self.entropy_coef * grad_entropy
        
        # Parameter update: θ ← θ + α·ēλ
        idx = 0
        for param in self.policy.parameters():
            num_params = param.numel()
            param.data += (self.learning_rate * 
                          total_eligibility[idx:idx+num_params].reshape(param.shape))
            idx += num_params
        
        return total_eligibility.norm().item()
    
    return None
```

---

## **Summary: Algorithm Flow**

```
1. Initialize policy πθ(a|s), buffer E, window w
2. For each episode:
   3. For each timestep t:
      4. Sample action: at ~ πθ(·|st)
      5. Execute action, observe st+1
      6. Record (st-d, at-d, pt-d) in window w
      7. If human gives feedback ft ≠ 0:
         8. Truncate w to last L entries
         9. Store (w, ft) in buffer E
         10. Reset w ← ∅
      11. Sample minibatch W from E
      12. For each (w, F) in W:
          13. Initialize eλ ← 0
          14. For each (s, a, p) in w:
              15. Compute importance ratio: πθ(a|s)/p
              16. Update: eλ ← λ·eλ + (πθ/p)·∇θ log πθ(a|s)
          17. Accumulate: ēλ ← ēλ + F·eλ
      18. Average: ēλ ← (1/m)·ēλ
      19. Add entropy: ēλ ← ēλ + β·∇θ H(πθ)
      20. Update: θ ← θ + α·ēλ
```

**Key Equations:**
- **Eligibility trace**: $e_\lambda \leftarrow \lambda \cdot e_\lambda + \frac{\pi_\theta(a|s)}{p} \cdot \nabla_\theta \log \pi_\theta(a|s)$
- **Parameter update**: $\theta \leftarrow \theta + \alpha \cdot \bar{e}_\lambda$
- **Entropy regularization**: $H(\pi_\theta) = \frac{1}{2}(1 + \log(2\pi)) + \log \sigma_\theta$

---

