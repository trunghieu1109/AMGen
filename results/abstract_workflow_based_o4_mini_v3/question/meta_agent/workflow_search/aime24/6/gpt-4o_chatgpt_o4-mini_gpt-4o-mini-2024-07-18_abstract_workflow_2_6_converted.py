async def forward_6(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Express the surface area condition as an equation in terms of the dimensions of a rectangular box (length, width, height)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, considering surface area condition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Express the volume condition as an equation in terms of the dimensions of a rectangular box (length, width, height)."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}

    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, volume condition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2

    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction = "Sub-task 3: Find the possible real values of box dimensions (length, width, height) satisfying both the surface area and volume conditions."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round

    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, resolving box dimensions, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Validate box dimensions resolution and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback on box dimensions, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining box dimensions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_4 = "Sub-task 4: Compute the radius of the smallest sphere that can contain these boxes for given dimensions."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating sphere radius, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Decide on sphere radius estimation.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding sphere radius, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = "Sub-task 5: Express the square of the radius calculated in subtask 4 as a fraction in its simplest form."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion Agent {cot_agent_5.id}, simplifying radius squared, thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Review fraction simplification.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback on radius squared simplification, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion Agent {cot_agent_5.id}, refining radius squared simplification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_6 = "Sub-task 6: Compute the sum of the numerator and denominator of the simplified fraction obtained in subtask 5."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    all_thinking6 = [[] for _ in range(N_max_4)]
    all_answer6 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for agent in debate_agents_6:
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, summing numerator and denominator, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Decide on sum computation.", is_sub_task=True)
    agents.append(f"Final Decision agent, summing numerator and denominator, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer