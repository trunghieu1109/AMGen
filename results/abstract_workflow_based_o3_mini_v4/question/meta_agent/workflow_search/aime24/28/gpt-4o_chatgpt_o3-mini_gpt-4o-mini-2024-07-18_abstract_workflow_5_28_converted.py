async def forward_28(self, taskInfo):
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Identify essential properties of the torus
    cot_instruction_1 = "Sub-task 1: Identify the essential geometric properties of the torus, including the major and minor radii."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determine geometric properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Understand tangency conditions at the inner circle
    cot_instruction_2 = "Sub-task 2: Understand the geometric configuration and tangency conditions when the torus is externally tangent to the sphere at its inner circle."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyze inner tangency conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Understand tangency conditions at the outer circle
    cot_instruction_3 = "Sub-task 3: Understand the geometric configuration and tangency conditions when the torus is externally tangent to the sphere at its outer circle."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyze outer tangency conditions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate the radii of the circles of tangency
    reflex_instruction_4 = "Sub-task 4: Calculate the radius of the circle of tangency when the torus is internally tangent to the sphere."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2, answer2]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflex_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, calculating internal tangency radius, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max_4):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "please review the internal tangency radius calculation.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflex_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining internal tangency radius, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    reflex_instruction_5 = "Sub-task 5: Calculate the radius of the circle of tangency when the torus is externally tangent to the sphere."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], reflex_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, calculating external tangency radius, thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max_4):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], "please review the external tangency radius calculation.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5 = [taskInfo, thinking5, answer5, feedback]
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflex_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining external tangency radius, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 3: Determine difference r_i - r_o
    cot_instruction_6 = "Sub-task 6: Determine the difference r_i - r_o based on calculated radii of tangency."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating difference r_i - r_o, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Stage 4: Express the difference in simplified form and calculate m+n
    debate_instruction_7 = "Sub-task 7: Express the difference r_i - r_o as a fraction m/n and find m+n ensuring m and n are coprime."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            input_infos_7 = [taskInfo, thinking6, answer6] if r == 0 else [taskInfo] + all_thinking7[r-1] + all_answer7[r-1]
            thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, expressing difference as fraction, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on m+n.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating m+n, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer