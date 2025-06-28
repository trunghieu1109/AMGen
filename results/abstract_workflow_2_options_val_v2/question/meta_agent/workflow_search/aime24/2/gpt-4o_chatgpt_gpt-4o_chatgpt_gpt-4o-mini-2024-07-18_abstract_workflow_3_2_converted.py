async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Sub-task 1: Calculate the total number of possible colorings of the octagon.
    cot_instruction = "Sub-task 1: Calculate the total number of possible colorings of the octagon, considering each vertex can be either red or blue."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculate total colorings, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the symmetries of the octagon.
    cot_sc_instruction = "Sub-task 2: Identify the symmetries of the octagon that allow for rotations, specifically the 8 rotational symmetries."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.7) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, identify symmetries, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Determine the number of colorings invariant under each symmetry.
    cot_reflect_instruction = "Sub-task 3: Determine the number of colorings that are invariant under each of the 8 rotational symmetries."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, determine invariant colorings, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the invariant colorings and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining invariant colorings, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Filter the colorings to find valid rotations.
    cot_reflect_instruction = "Sub-task 4: Filter the colorings to find those that can be rotated such that all blue vertices end up at positions where there were originally red vertices."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, filter valid rotations, thinking: {thinking4.content}; answer: {answer4.content}')
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], "please review the valid rotations filtering and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining valid rotations, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate the probability of a valid coloring.
    cot_reflect_instruction = "Sub-task 5: Calculate the probability of a valid coloring occurring by dividing the number of valid colorings by the total number of colorings."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculate probability, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking5, answer5], "please review the probability calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining probability calculation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Express the probability as a fraction and find m+n.
    debate_instruction_6 = "Sub-task 6: Express the probability as a fraction in simplest form and find the sum of the numerator and denominator."
    debate_agents_6 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = agent([taskInfo, thinking5, answer5], debate_instruction_6, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = agent(input_infos_6, debate_instruction_6, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, express probability as fraction, thinking: {thinking6.content}; answer: {answer6.content}')
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on m+n.", is_sub_task=True)
    agents.append(f'Final Decision agent, calculate m+n, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer