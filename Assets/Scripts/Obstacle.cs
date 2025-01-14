using UnityEngine;

public class Obstacle : MonoBehaviour
{
    public float bonusNegativeReward = -10f;
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Agent"))
        {
            Agent agent = other.GetComponent<Agent>();
            if (agent != null && !agent.done)
            {
                agent.ReceiveBonusReward(bonusNegativeReward);
            }
        }
    }
}
