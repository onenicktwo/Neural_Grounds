using UnityEngine;

public class Goal : MonoBehaviour
{
    public float bonusRewardValue = 10f;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Agent"))
        {
            Agent agent = other.GetComponent<Agent>();
            if (agent != null && !agent.hasReachedGoal)
            {
                agent.ReceiveBonusReward(bonusRewardValue);
                agent.hasReachedGoal = true; 
            }
        }
    }
}