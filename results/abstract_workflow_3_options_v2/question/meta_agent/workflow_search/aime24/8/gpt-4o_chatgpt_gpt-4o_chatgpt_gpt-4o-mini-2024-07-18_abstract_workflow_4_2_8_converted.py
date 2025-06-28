async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Core Information and Constraint Extraction
    # Sub-task 1: Identify the rules of the game and the possible moves each player can make.
    cot_instruction_1 = "Sub-task 1: Identify the rules of the game and the possible moves each player can make."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine game rules, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine the winning and losing positions for the game based on the rules identified.
    cot_instruction_2 = "Sub-task 2: Determine the winning and losing positions for the game based on the rules identified."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determine winning/losing positions, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Parametric Representation Derivation
    # Sub-task 3: Identify the parametric representation of winning and losing positions.
    cot_instruction_3 = "Sub-task 3: Identify the parametric representation of winning and losing positions."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, identify parametric representation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine the conditions under which Bob can force a win.
    cot_sc_instruction_4 = "Sub-task 4: Determine the conditions under which Bob can force a win."
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_4[i].id}, determine conditions for Bob to win, thinking: {thinking4.content}; answer: {answer4.content}')
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Expression Transformation and Alignment
    # Sub-task 5: Transform the conditions into a mathematical expression or pattern.
    cot_reflect_instruction_5 = "Sub-task 5: Transform the conditions into a mathematical expression or pattern."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, transform conditions, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "please review the transformation and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refine transformation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Stage 4: Final Output Calculation
    # Sub-task 6: Calculate the number of positive integers n less than or equal to 2024 for which Bob can guarantee a win.
    debate_instruction_6 = "Sub-task 6: Calculate the number of positive integers n less than or equal to 2024 for which Bob can guarantee a win."
    debate_agents_6 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
            thinking6, answer6 = agent(input_infos_6, debate_instruction_6, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating number of positions, thinking: {thinking6.content}; answer: {answer6.content}')
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make a final decision on the number of positions.", is_sub_task=True)
    agents.append(f'Final Decision agent on calculating number of positions, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer