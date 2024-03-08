using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class MultirollerAgent : Agent
{
    public Transform otherAgent;
    public Rigidbody2D plank;
    public BoxCollider2D plankCollider;
    public PolygonCollider2D floorCollider;
    public EnvironmentManager environmentManager;

    private Rigidbody2D agentBody;
    private CircleCollider2D agentCollider;
    private float lastDistance;

    void Start()
    {
        agentBody = GetComponent<Rigidbody2D>();
        agentCollider = GetComponent<CircleCollider2D>();
    }

    public override void OnEpisodeBegin()
    {
        environmentManager.ResetEnvironmentIfNeeded();
        lastDistance = Parameters.initPlankPosition.x;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float plankRotation = plank.rotation * Mathf.Deg2Rad;
        Vector3 agentDistance = otherAgent.localPosition - this.transform.localPosition;
        Vector3 plankDistance = plank.transform.localPosition - this.transform.localPosition;
        sensor.AddObservation(agentDistance.x);
        sensor.AddObservation(agentDistance.y);
        sensor.AddObservation(plankDistance.x);
        sensor.AddObservation(plankDistance.y);
        sensor.AddObservation(Mathf.Cos(plankRotation));
        sensor.AddObservation(Mathf.Sin(plankRotation));
        sensor.AddObservation(Physics2D.Distance(agentCollider, floorCollider).distance);
        sensor.AddObservation(Physics2D.Distance(plankCollider, floorCollider).distance);
        sensor.AddObservation(Parameters.goalPositionX - plank.transform.localPosition.x);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        Debug.Log("RUNNING");
        Vector2 controlSignal = Vector2.zero;
        var discreteActions = actions.DiscreteActions;
        int action = discreteActions[0];

        controlSignal.y = agentBody.velocity.y;
        if (action == 0) // Stop
        {
            controlSignal.x = 0;
        }
        else if (action == 1) // Go Left
        {
            controlSignal.x = -Parameters.speed;
        }
        else if (action == 2) // Go Right
        {
            controlSignal.x = Parameters.speed;
        }
        else if (action == 3) // Jump
        {
            controlSignal.x = 0;
            if (Physics2D.Distance(agentCollider, floorCollider).distance < 0.002f)
            {
                controlSignal.y = Parameters.jumpSpeed;
            }
        }
        agentBody.velocity = controlSignal;

        float reward = plank.position.x - lastDistance;
        lastDistance = plank.position.x;

        //Debug.Log(ColliderDistance.IsColliding(plankCollider, floorCollider));
        if (Physics2D.Distance(plankCollider, floorCollider).distance < 0.002f || plank.position.x <= Parameters.backwardsLimitX)
        {
            SetReward(-100f);
            EndEpisode_();
        }
        else
        {
            SetReward(reward*10);
            if (plank.position.x >= Parameters.goalPositionX)
            {
                EndEpisode_();
            }
        }
    }

    public void EndEpisode_()
    {
        environmentManager.agentCompleted();
        EndEpisode();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = Random.Range(0, 4);
    }
}
