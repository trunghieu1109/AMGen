async def forward_29(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Identify conditions for maximum chips per row/column.

    # Subtask 1: Max chips per row without violating conditions.
    cot_instruction_1 = "Sub-task 1: Determine the maximum number of chips that can be placed in a single row, with each row containing chips of the same color."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining max chips per row, thinking: {thinking1.content}; answer: {answer1.content}")

    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Subtask 2: Max chips per column, same condition applies
    cot_instruction_2 = "Sub-task 2: Determine the maximum number of chips that can be placed in a single column, with each column containing chips of the same color."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determining max chips per column, thinking: {thinking2.content}; answer: {answer2.content}")

    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Subtask 3: Potential row configurations using results of Subtask 1
    cot_reflection_instruction_3 = "Sub-task 3: Considering the result from Sub-task 1, determine potential configurations of chip placements across rows ensuring independent rows."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_reflection_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, potential row configurations, thinking: {thinking3.content}; answer: {answer3.content}")

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Subtask 4: Potential column configurations using results of Subtask 2
    cot_reflection_instruction_4 = "Sub-task 4: Considering the result from Sub-task 2, determine potential configurations of chip placements across columns ensuring independent columns."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_reflection_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, potential column configurations, thinking: {thinking4.content}; answer: {answer4.content}")

    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Combine configurations and validate

    # Sub-task 5: Synthesize row and column configurations
    debate_instruction_5 = "Sub-task 5: Synthesize row configurations and column configurations to assess compatible grid placements."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]

    for r in range(self.max_round):
        for agent in debate_agents_5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, synthesizing grid placements, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on synthesized grid placements.", is_sub_task=True)
    agents.append(f"Final Decision agent, assessing grid placements, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Validate potential grid configurations
    cot_reflect_instruction_6 = "Sub-task 6: Validate potential configurations ensuring total chips don't exceed collection and conditions hold."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    N_max = self.max_round

    cot_inputs = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, validating configurations, thinking: {thinking6.content}; answer: {answer6.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please ensure the validation of configurations and check its accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, reviewing configuration validation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining configurations validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer