async def forward_23(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1, Subtask 1: Determine two numbers summing up to 999 from each row.
    cot_instruction_1 = "Sub-task 1: Identify all pairs of numbers that can be formed from a 2x3 grid such that their sum equals 999 when read left to right."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying pairs for 999 sum: thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1, Subtask 2: Determine sets of three numbers summing up to 99 from top to bottom.
    cot_sc_instruction_2 = "Sub-task 2: Identify sets of three numbers that can be formed in the grid such that they sum to 99, considering the vertical column constraints."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying sets for 99 sum: thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2, Subtask 3: Align conditions to find potential grid configurations.
    cot_reflect_instruction_3 = "Sub-task 3: Align the conditions from previous subtasks to identify feasible 2x3 grid configurations."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, aligning conditions: thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "Review the grid configurations alignment and provide feedback on its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback: thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining grid configurations: thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2, Subtask 4: Validate configurations to ensure all conditions are satisfied and find final result.
    debate_instruction_4 = "Sub-task 4: Validate potential grid configurations to find final valid results satisfying all conditions."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]

            if r > 0:
                input_infos_4.extend(all_thinking_4[r-1])

            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating configurations: thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make the final decision on grid configuration validation.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining final configuration: thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
