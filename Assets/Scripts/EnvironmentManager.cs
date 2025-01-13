using UnityEngine;
using System.Collections.Generic;

public class EnvironmentManager : MonoBehaviour
{
    public static EnvironmentManager Instance { get; private set; }

    public GameObject agentPrefab;
    public GameObject goalPrefab;
    public int gridSize = 10;
    public int numberOfAgents = 10;

    [HideInInspector]
    public Vector3 goalPosition;

    private List<Agent> agents = new List<Agent>();
    private GameObject goalObject;

    public bool IsInitialized { get; private set; }

    private int episodeCount = 0;
    private int stepsInEpisode = 0;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void Start()
    {
        InitializeEnvironment();
    }

    private void InitializeEnvironment()
    {
        // Place goal
        goalObject = Instantiate(goalPrefab, Vector3.zero, Quaternion.identity);
        SetRandomGoalPosition();

        // Place agents
        Vector3 agentStartPos = Vector3.zero;
        for (int i = 0; i < numberOfAgents; i++)
        {
            GameObject agentObj = Instantiate(agentPrefab, agentStartPos, Quaternion.identity);
            Agent agent = agentObj.GetComponent<Agent>();
            agents.Add(agent);
        }

        IsInitialized = true;
    }

    private void SetRandomGoalPosition()
    {
        Vector3 newGoalPos = new Vector3(Random.Range(0, gridSize), 0, Random.Range(0, gridSize));
        goalObject.transform.position = newGoalPos;
        goalPosition = newGoalPos;
    }

    private void EndEpisode()
    {
        episodeCount++;
        float totalReward = 0f;
        foreach (Agent agent in agents)
        {
            totalReward += agent.GetEpisodeReward();
        }

        Debug.Log($"Episode {episodeCount} completed in {stepsInEpisode} steps. Average reward per agent: {totalReward / agents.Count}");

        TrainingManager.Instance.IncrementEpisode();
    }

    public void ResetEnvironment()
    {
        SetRandomGoalPosition();

        foreach (Agent agent in agents)
        {
            agent.ResetPosition();
        }

        stepsInEpisode = 0;
    }

    public void StepAgents()
    {
        bool allAgentsReachedGoal = true;

        foreach (Agent agent in agents)
        {
            if (!agent.hasReachedGoal)
            {
                agent.Act();
                allAgentsReachedGoal = false; 
            }
        }

        stepsInEpisode++;

        if (allAgentsReachedGoal)
        {
            Debug.Log("All agents have reached the goal, ending episode early.");
            EndEpisode();
            return; 
        }

        if (stepsInEpisode >= TrainingManager.Instance.EpisodeLength)
        {
            EndEpisode();
        }
    }
}