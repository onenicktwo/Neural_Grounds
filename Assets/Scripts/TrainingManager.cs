using UnityEngine;

public class TrainingManager : MonoBehaviour
{
    public static TrainingManager Instance { get; private set; }

    public int EpisodeLength = 1000;
    public int TotalEpisodes = 100;

    public int CurrentStep;
    public int CurrentEpisode;

    private bool isTraining = false;

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
        StartCoroutine(WaitForInitializationAndStartTraining());
    }

    private System.Collections.IEnumerator WaitForInitializationAndStartTraining()
    {
        while (!EnvironmentManager.Instance.IsInitialized)
        {
            yield return null;
        }

        StartTraining();
    }

    private void StartTraining()
    {
        CurrentEpisode = 0;
        StartNewEpisode();
        isTraining = true;
    }

    private void StartNewEpisode()
    {
        CurrentStep = 0;
        EnvironmentManager.Instance.ResetEnvironment();
    }

    private void Update()
    {
        if (isTraining && CurrentEpisode < TotalEpisodes)
        {
            if (CurrentStep < EpisodeLength)
            {
                EnvironmentManager.Instance.StepAgents();
                CurrentStep++;
            }
            else
            {
                CurrentEpisode++;
                if (CurrentEpisode < TotalEpisodes)
                {
                    StartNewEpisode();
                }
                else
                {
                    Debug.Log("Training completed!");
                    isTraining = false;
                }
            }
        }
    }

    public void IncrementEpisode()
    {
        CurrentEpisode++;
        StartNewEpisode();
    }
}