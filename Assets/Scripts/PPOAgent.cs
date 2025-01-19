using UnityEngine;
using System.Collections.Generic;

public class PPOAgent : MonoBehaviour
{
    // Hyperparameters
    public float learningRate = 0.01f;
    public float gamma = 0.99f; // Discount factor
    public float clipEpsilon = 0.2f;
    public int rolloutLength = 128; // Number of steps per training batch
    public int epochs = 4;

    // Agent movement
    public float moveSpeed = 2f;

    // Model parameters (weights for simplicity)
    private float[] policyWeights; // Policy network weights
    private float[] valueWeights;  // Value network weights

    // Rollout data
    private List<float[]> states = new List<float[]>(); // State history
    private List<int> actions = new List<int>();        // Action history
    private List<float> rewards = new List<float>();    // Reward history
    private List<float> values = new List<float>();     // Value estimations
    private List<float> advantages = new List<float>(); // Advantage estimations

    // Unity-specific components
    private Rigidbody rb;

    private void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Initialize policy and value weights randomly
        policyWeights = new float[4]; // Assuming 4 input features
        valueWeights = new float[4];
        for (int i = 0; i < 4; i++)
        {
            policyWeights[i] = Random.Range(-0.1f, 0.1f);
            valueWeights[i] = Random.Range(-0.1f, 0.1f);
        }
    }

    private void FixedUpdate()
    {
        // Get the current state
        float[] state = GetState();

        // Choose an action using the policy
        int action = ChooseAction(state);

        // Perform the action
        PerformAction(action);

        // Calculate reward and store rollout data
        float reward = CalculateReward();
        states.Add(state);
        actions.Add(action);
        rewards.Add(reward);

        // Check if episode is over
        if (IsEpisodeOver())
        {
            // Perform PPO update
            Train();

            // Reset the environment
            ResetEnvironment();
        }
    }

    private float[] GetState()
    {
        // Example state: agent's position relative to the goal
        Vector3 agentPos = transform.position;
        Vector3 goalPos = GameObject.Find("Goal").transform.position;
        return new float[] { agentPos.x, agentPos.z, goalPos.x, goalPos.z };
    }

    private int ChooseAction(float[] state)
    {
        // Softmax policy
        float[] logits = Forward(policyWeights, state);
        float[] probabilities = Softmax(logits);

        // Sample an action based on probabilities
        float rand = Random.value;
        float cumulative = 0f;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (rand < cumulative)
                return i;
        }
        return probabilities.Length - 1; // Fallback action
    }

    private void PerformAction(int action)
    {
        // Actions: 0 = forward, 1 = backward, 2 = left, 3 = right
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
        // Reward function: +1 for reaching the goal, -0.01 for each step, -1 for hitting obstacles
        Vector3 agentPos = transform.position;
        Vector3 goalPos = GameObject.Find("Goal").transform.position;

        if (Vector3.Distance(agentPos, goalPos) < 1f)
        {
            return 1f; // Goal reached
        }
        if (Physics.CheckSphere(agentPos, 0.5f, LayerMask.GetMask("Obstacle")))
        {
            return -1f; // Hit obstacle
        }
        return -0.01f; // Small penalty for each step
    }

    private bool IsEpisodeOver()
    {
        // End the episode if the agent reaches the goal or a maximum number of steps is reached
        Vector3 agentPos = transform.position;
        Vector3 goalPos = GameObject.Find("Goal").transform.position;
        return Vector3.Distance(agentPos, goalPos) < 1f || rewards.Count >= 1000;
    }

    private void Train()
    {
        // Calculate advantages and discounted rewards
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
                float newProbability = Mathf.Exp(logits[action]); // Recalculate with updated weights
                float ratio = newProbability / oldProbability;

                // Clip the probability ratio
                float clippedRatio = Mathf.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon);
                float policyLoss = -Mathf.Min(ratio * advantage, clippedRatio * advantage);

                // Update policy weights (gradient descent)
                for (int j = 0; j < policyWeights.Length; j++)
                {
                    policyWeights[j] -= learningRate * policyLoss * state[j];
                }

                // Update value network
                float targetValue = rewards[i] + gamma * (i < rewards.Count - 1 ? values[i + 1] : 0);
                float valueLoss = Mathf.Pow(targetValue - values[i], 2);
                for (int j = 0; j < valueWeights.Length; j++)
                {
                    valueWeights[j] -= learningRate * valueLoss * state[j];
                }
            }
        }

        // Clear rollout data
        states.Clear();
        actions.Clear();
        rewards.Clear();
        values.Clear();
        advantages.Clear();
    }

    private void CalculateAdvantages()
    {
        float lastValue = 0f;
        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            float tdError = rewards[i] + gamma * lastValue - values[i];
            advantages.Insert(0, tdError);
            lastValue = values[i];
        }
    }

    private float[] Forward(float[] weights, float[] input)
    {
        // Simple linear model: weights * input
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
        // Convert logits to probabilities using softmax
        float maxLogit = Mathf.Max(logits);
        float sum = 0f;
        float[] probabilities = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = Mathf.Exp(logits[i] - maxLogit); // Normalize logits for numerical stability
            sum += probabilities[i];
        }
        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] /= sum; // Divide by the sum to get probabilities
        }
        return probabilities;
    }

    private void ResetEnvironment()
    {
        // Reset agent position and goal position
        transform.position = new Vector3(0, 0.5f, 0); // Reset agent to starting position
        rb.velocity = Vector3.zero; // Stop any movement

        GameObject goal = GameObject.Find("Goal");
        goal.transform.position = new Vector3(
            Random.Range(-5f, 5f),
            0.5f,
            Random.Range(-5f, 5f)
        );

        // Clear rollout data for the next episode
        states.Clear();
        actions.Clear();
        rewards.Clear();
        values.Clear();
        advantages.Clear();
    }
}