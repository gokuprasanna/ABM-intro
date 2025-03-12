def run_simulation():
    model = LLMModel(num_agents=5, width=10, height=10)
    for i in range(10):  # Run for 10 steps
        model.step()
        for agent in model.schedule.agents:
            response = agent.generate_response("What's your current state?")
            print(f"Agent {agent.unique_id}: {response}")

if __name__ == "__main__":
    run_simulation()
