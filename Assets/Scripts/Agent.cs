using UnityEngine;
using System.Collections.Generic;

public class Agent : MonoBehaviour
{
    public float learningRate = 0.1f;
    public float discountFactor = 0.95f;
    public float explorationRate = 0.1f;
    public float moveSpeed = 1f;

    public bool visualizePath = true;
    public float pathPointInterval = 0.1f;
    public Color pathColor = Color.red;

    private Dictionary<string, float> qTable = new Dictionary<string, float>();
    private Vector3 startPosition;
    private Vector3 previousPosition;
    private float accumulatedReward = 0f;
    private List<Vector3> pathPoints = new List<Vector3>();
    private float lastPathPointTime;

    public bool hasReachedGoal = false;

    private float episodeReward = 0f;

    private void Start()
    {
        startPosition = transform.position;
        previousPosition = startPosition;
    }

    private void Update()
    {
        if (visualizePath && Time.time - lastPathPointTime > pathPointInterval)
        {
            pathPoints.Add(transform.position);
            lastPathPointTime = Time.time;
        }
    }

    public void Act()
    {
        if (hasReachedGoal) return;

        string state = GetCurrentState();
        string action = ChooseAction(state);
        PerformAction(action);
        float reward = GetReward();
        string nextState = GetCurrentState();
        UpdateQTable(state, action, reward, nextState);
        episodeReward += reward;

        if (visualizePath && Time.time - lastPathPointTime > pathPointInterval)
        {
            pathPoints.Add(transform.position);
            lastPathPointTime = Time.time;
        }
    }

    private string ChooseAction(string state)
    {
        if (Random.value < explorationRate)
        {
            return GetRandomAction();
        }
        else
        {
            return GetBestAction(state);
        }
    }

    private void UpdateQTable(string state, string action, float reward, string nextState)
    {
        string stateAction = state + action;
        float oldValue = qTable.ContainsKey(stateAction) ? qTable[stateAction] : 0f;
        float nextMax = GetMaxQValue(nextState);

        float newValue = (1 - learningRate) * oldValue +
                         learningRate * (reward + discountFactor * nextMax);

        qTable[stateAction] = newValue;
    }

    private string GetCurrentState()
    {
        int x = Mathf.RoundToInt(transform.position.x);
        int z = Mathf.RoundToInt(transform.position.z);
        return $"{x},{z}";
    }

    private string GetRandomAction()
    {
        string[] actions = { "up", "down", "left", "right" };
        return actions[Random.Range(0, actions.Length)];
    }

    private string GetBestAction(string state)
    {
        string[] actions = { "up", "down", "left", "right" };
        string bestAction = GetRandomAction();
        float bestValue = float.MinValue;

        foreach (string action in actions)
        {
            string stateAction = state + action;
            float value = qTable.ContainsKey(stateAction) ? qTable[stateAction] : 0f;
            if (value > bestValue)
            {
                bestValue = value;
                bestAction = action;
            }
        }

        return bestAction;
    }

    private void PerformAction(string action)
    {
        Vector3 movement = Vector3.zero;
        switch (action)
        {
            case "up": movement = Vector3.forward; break;
            case "down": movement = Vector3.back; break;
            case "left": movement = Vector3.left; break;
            case "right": movement = Vector3.right; break;
        }
        previousPosition = transform.position;
        transform.position += movement * moveSpeed * Time.deltaTime;
    }

    private float GetReward()
    {
        float previousDistance = Vector3.Distance(previousPosition, EnvironmentManager.Instance.goalPosition);
        float currentDistance = Vector3.Distance(transform.position, EnvironmentManager.Instance.goalPosition);

        float reward = previousDistance - currentDistance;

        reward -= 0.01f;

        reward += accumulatedReward;
        accumulatedReward = 0f;

        return reward;
    }

    private float GetMaxQValue(string state)
    {
        string[] actions = { "up", "down", "left", "right" };
        float maxValue = float.MinValue;

        foreach (string action in actions)
        {
            string stateAction = state + action;
            float value = qTable.ContainsKey(stateAction) ? qTable[stateAction] : 0f;
            if (value > maxValue)
            {
                maxValue = value;
            }
        }

        return maxValue;
    }

    public void ResetPosition()
    {
        transform.position = startPosition;
        previousPosition = startPosition;
        pathPoints.Clear();
        lastPathPointTime = 0f;
        accumulatedReward = 0f;
        episodeReward = 0f;
        hasReachedGoal = false;
    }

    public void ReceiveBonusReward(float bonusReward)
    {
        accumulatedReward += bonusReward;
    }

    private void OnDrawGizmos()
    {
        if (visualizePath && pathPoints.Count > 1)
        {
            Gizmos.color = pathColor;
            for (int i = 1; i < pathPoints.Count; i++)
            {
                Gizmos.DrawLine(pathPoints[i - 1], pathPoints[i]);
            }
        }
    }

    public float GetEpisodeReward()
    {
        return episodeReward;
    }
}