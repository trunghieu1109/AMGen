async def forward_2(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Stage 1
    cot_instruction = "Determine the total number of ways to color the vertices of the octagon with two colors."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determining ways to color the octagon, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2
    cot_sc_instruction = "Identify the rotational symmetries of the octagon that preserve the color configuration."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying rotational symmetries, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2

    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3
    cot_reflect_instruction = "Determine under what conditions the octagon's colored vertices can be rotated."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, determining rotation conditions, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "please review the rotation conditions.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining rotation conditions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4
    cot_debate_instruction = "Calculate the number of colorings that satisfy the conditions in Sub-task 3."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_debate = self.max_round
    all_thinking = [[] for _ in range(N_max_debate)]
    all_answer = [[] for _ in range(N_max_debate)]
    
    for r in range(N_max_debate):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], cot_debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking[r-1] + all_answer[r-1]
                thinking4, answer4 = await agent(input_infos, cot_debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating valid colorings, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking[r].append(thinking4)
            all_answer[r].append(answer4)

    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 4: Finalize the number of valid colorings.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining valid colorings, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Stage 2
    cot_instruction_5 = "Calculate the probability that the coloring allows for a rotation meeting conditions in Sub-task 4."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating probability, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_6 = "Express the probability as a reduced fraction and find m+n."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, reducing fraction, thinking: {thinking6.content}; answer: {answer6.content}")

    for i in range(N_max_6):
        feedback_6, correct_6 = await critic_agent_6([taskInfo, thinking6, answer6], "please review the fraction reduction.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback_6.content}; answer: {correct_6.content}")
        if correct_6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback_6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining fraction, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer