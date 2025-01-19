using UnityEngine;
using System.Collections.Generic;

public class PPOAgent : MonoBehaviour
{
    // Hyperparameters
    public float learningRate = 0.01f; // Speed of learning, keep low for stability, (0, 1)
    public float gamma = 0.99f; // Discount factor, keep high so agent thinks future rewards are more important (0, 1)
    public float clipEpsilon = 0.2f; // Limits the size of policy updates in PPO, keep low for stability
    public int rolloutLength = 128; // Number of steps per training batch, high = stable, slow to adapt, low = quick, noisy
    public int epochs = 4; //# of times data is iterated over, high = thorough, possible overfit, low = might not extract all data

    // Agent movement
    public float moveSpeed = 2f;

    // Episode time limit
    public int maxEpisodeSteps = 500;
    [SerializeField]
    private int currentStep = 0;

    // Model parameters
    private float[] policyWeights; // Policy network weights
    private float[] valueWeights;  // Value network weights

    // Rollout data
    private List<float[]> states = new List<float[]>(); // State history
    private List<int> actions = new List<int>();        // Action history
    private List<float> rewards = new List<float>();    // Reward history
    private List<float> values = new List<float>();     // Value of state estimations
    private List<float> advantages = new List<float>(); // Advantage estimations

    private Rigidbody rb;

    private void Start()
    {
        rb = GetComponent<Rigidbody>();

        policyWeights = new float[4];
        valueWeights = new float[4];
        for (int i = 0; i < 4; i++)
        {
            policyWeights[i] = Random.Range(-0.1f, 0.1f);
            valueWeights[i] = Random.Range(-0.1f, 0.1f);
        }
    }

    private void FixedUpdate()
    {
        currentStep++;

        float[] state = GetState();
        float value = EstimateValue(state);
        int action = ChooseAction(state);

        PerformAction(action);

        float reward = CalculateReward();
        states.Add(state);
        actions.Add(action);
        rewards.Add(reward);
        values.Add(value);

        if (HasFailed() || HasSucceeded() || currentStep >= maxEpisodeSteps)
        {
            Train();
            ResetEnvironment();
        }
    }

    private float[] GetState()
    {
        Vector3 agentPos = transform.position;
        Vector3 goalPos = GameObject.Find("Goal").transform.position;
        return new float[] { agentPos.x, agentPos.z, goalPos.x, goalPos.z };
    }

    private int ChooseAction(float[] state)
    {
        float[] logits = Forward(policyWeights, state);
        float[] probabilities = Softmax(logits);

        float rand = Random.value;
        float cumulative = 0f;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (rand < cumulative)
                return i;
        }
        return probabilities.Length - 1;
    }

    private void PerformAction(int action)
    {
        Vector3 move = Vector3.zero;
        switch (action)
        {
            case 0: move = Vector3.forward; break;
            case 1: move = Vector3.back; break;
            case 2: move = Vector3.left; break;
            case 3: move = Vector3.right; break;
        }
        rb.MovePosition(rb.position + move * moveSpeed * Time.fixedDeltaTime);
    }

    private float CalculateReward()
    {
        Vector3 agentPos = transform.position;
        Vector3 goalPos = GameObject.Find("Goal").transform.position;

        if (HasSucceeded())
        {
            return 1f;
        }
        if (HasFailed())
        {
            return -1f; 
        }
        return -0.01f;
    }

    private bool HasSucceeded()
    {
        Vector3 agentPos = transform.position;
        Vector3 goalPos = GameObject.Find("Goal").transform.position;
        return Vector3.Distance(agentPos, goalPos) < 1f;
    }

    private bool HasFailed()
    {
        Vector3 agentPos = transform.position;
        return Physics.CheckSphere(agentPos, 0.5f, LayerMask.GetMask("Obstacle"));
    }

    private void Train()
    {
        CalculateAdvantages();

        // Update policy and value networks using PPO
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < states.Count; i++)
            {
                // Update policy
                float[] state = states[i];
                int action = actions[i];
                float advantage = advantages[i];

                // Calculate policy gradient
                float[] logits = Forward(policyWeights, state);
                float[] probabilities = Softmax(logits);
                float oldProbability = probabilities[action];
                float newProbability = Mathf.Exp(logits[action]);
                float ratio = newProbability / oldProbability; // Policy ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)

                // Clipped surrogate objective: L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
                float clippedRatio = Mathf.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon);
                float policyLoss = -Mathf.Min(ratio * advantage, clippedRatio * advantage);

                // Gradient descent update: θ = θ - α * ∇L
                for (int j = 0; j < policyWeights.Length; j++)
                {
                    policyWeights[j] -= learningRate * policyLoss * state[j];
                }

                // Value function update
                // TD target: y = r + γV(s')
                float targetValue = rewards[i] + gamma * (i < rewards.Count - 1 ? values[i + 1] : 0);
                // Mean Squared Error loss: L = (y - V(s))^2
                float valueLoss = Mathf.Pow(targetValue - values[i], 2);
                for (int j = 0; j < valueWeights.Length; j++)
                {
                    valueWeights[j] -= learningRate * valueLoss * state[j];
                }
            }
        }

        states.Clear();
        actions.Clear();
        rewards.Clear();
        values.Clear();
        advantages.Clear();
    }

    private void CalculateAdvantages()
    {
        float lastValue = 0f;
        advantages.Clear();
        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            // TD error as advantage estimate: A(s,a) ≈ r + γV(s') - V(s)
            float tdError = rewards[i] + gamma * lastValue - values[i];
            advantages.Insert(0, tdError);
            lastValue = values[i];
        }
    }

    private float EstimateValue(float[] state)
    {
        // Linear value function approximation: V(s) = w^T * s
        float value = 0f;
        for (int i = 0; i < state.Length; i++)
        {
            value += valueWeights[i] * state[i];
        }
        return value;
    }

    private float[] Forward(float[] weights, float[] input)
    {
        // Linear layer: output = weights * input
        float[] output = new float[4];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = 0f;
            for (int j = 0; j < input.Length; j++)
            {
                output[i] += weights[j] * input[j];
            }
        }
        return output;
    }

    private float[] Softmax(float[] logits)
    {
        // Softmax function: P(a|s) = exp(log_a) / Σ exp(log_i)
        float maxLogit = Mathf.Max(logits);
        float sum = 0f;
        float[] probabilities = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = Mathf.Exp(logits[i] - maxLogit);
            sum += probabilities[i];
        }
        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] /= sum;
        }
        return probabilities;
    }

    private void ResetEnvironment()
    {
        transform.position = new Vector3(-5f, 0, 0);
        rb.velocity = Vector3.zero;

        currentStep = 0;

        states.Clear();
        actions.Clear();
        rewards.Clear();
        values.Clear();
        advantages.Clear();
    }
}