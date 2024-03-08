using UnityEngine;

public class EnvironmentManager : MonoBehaviour
{
    public Rigidbody2D agent1;
    public Rigidbody2D agent2;
    public Rigidbody2D plank;

    private int agentsFinished = 0;

    public void ResetEnvironmentIfNeeded()
    {
        if (agentsFinished >= 2)
        {
            agentsFinished = 0;

            // reset agent1
            agent1.angularVelocity = 0;
            agent1.velocity = Vector2.zero;
            agent1.transform.localPosition = Parameters.initAgent1Position;

            // reset agent2
            agent2.angularVelocity = 0;
            agent2.velocity = Vector2.zero;
            agent2.transform.localPosition = Parameters.initAgent2Position;

            // reset plank
            plank.rotation = 0;
            plank.angularVelocity = 0;
            plank.velocity = Vector2.zero;
            plank.transform.localPosition = Parameters.initPlankPosition;
        }
    }

    public void agentCompleted()
    {
        agentsFinished += 1;
    }

}
