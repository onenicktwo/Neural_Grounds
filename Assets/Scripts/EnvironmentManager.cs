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
        goalObject = Instantiate(goalPrefab, Vector3.zero, Quaternion.identity);
        SetRandomGoalPosition();

        Vector3 agentStartPos = new Vector3(-8f, 0, 5f);
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
        bool allAgentsDone = true;

        foreach (Agent agent in agents)
        {
            if (!agent.done)
            {
                agent.Act();
                allAgentsDone = false; 
            }
        }

        stepsInEpisode++;

        if (allAgentsDone)
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