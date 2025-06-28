async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction_1 = "Sub-task 1: Identify and classify the physical process involving gamma-ray annihilation with a CMB photon and extract the given parameters."
    cot_agent_1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, identifying process and parameters, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Recall and state the relativistic threshold condition for electron-positron pair production in gamma-gamma collisions in the lab frame."
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    for i in range(N_sc):
        thinking2_i, answer2_i = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, recalling threshold condition, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers2.append(answer2_i.content)
        thinking_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    most_common2 = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[most_common2]
    answer2 = answer_map2[most_common2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3 = "Sub-task 3: Derive the general expression for the gamma-ray threshold energy E_thr in terms of the target photon energy ε and angular factor (1-cos θ)."
    cot_agent_3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_reflect = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, deriving general threshold formula, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_reflect):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "Critically evaluate the derived general expression for correctness and completeness.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, feedback on general expression, thinking: {feedback3.content}; answer: {correct3.content}')
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining general formula, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = "Sub-task 4: Generate two variants of the threshold formula for (1-cos θ) = 2 (head-on) and = 1 (alignment)."
    cot_agent_4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, generating formula variants, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Compute (m_e c^2)^2 in eV^2 using m_e c^2 = 5.11e5 eV."
    cot_agent_5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, computing (m_e c^2)^2, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = "Sub-task 6: Using ε = 1e-3 eV and (1-cos θ)=1, compute E_thr1 = (m_e c^2)^2 / (ε * 1)."
    cot_agents_6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers6 = []
    thinking_map6 = {}
    answer_map6 = {}
    for i in range(N_sc):
        thinking6_i, answer6_i = await cot_agents_6[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_6[i].id}, computing E_thr1, thinking: {thinking6_i.content}; answer: {answer6_i.content}')
        possible_answers6.append(answer6_i.content)
        thinking_map6[answer6_i.content] = thinking6_i
        answer_map6[answer6_i.content] = answer6_i
    most_common6 = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinking_map6[most_common6]
    answer6 = answer_map6[most_common6]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction_7 = "Sub-task 7: Using ε = 1e-3 eV and (1-cos θ)=2, compute E_thr2 = (m_e c^2)^2 / (ε * 2)."
    cot_agents_7 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers7 = []
    thinking_map7 = {}
    answer_map7 = {}
    for i in range(N_sc):
        thinking7_i, answer7_i = await cot_agents_7[i]([taskInfo, thinking5, answer5], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_7[i].id}, computing E_thr2, thinking: {thinking7_i.content}; answer: {answer7_i.content}')
        possible_answers7.append(answer7_i.content)
        thinking_map7[answer7_i.content] = thinking7_i
        answer_map7[answer7_i.content] = answer7_i
    most_common7 = Counter(possible_answers7).most_common(1)[0][0]
    thinking7 = thinking_map7[most_common7]
    answer7 = answer_map7[most_common7]
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    debate_instruction_8 = "Sub-task 8: Convert E_thr1 and E_thr2 from eV to GeV and compare with the provided multiple-choice options to select the best match."
    debate_agents_8 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round = self.max_round
    all_thinking8 = [[] for _ in range(N_round)]
    all_answer8 = [[] for _ in range(N_round)]
    for r in range(N_round):
        for i, agent in enumerate(debate_agents_8):
            input_infos = [taskInfo, thinking6, answer6, thinking7, answer7]
            if r > 0:
                input_infos += all_thinking8[r-1] + all_answer8[r-1]
            thinking8_i, answer8_i = await agent(input_infos, debate_instruction_8, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, converting and comparing, thinking: {thinking8_i.content}; answer: {answer8_i.content}')
            all_thinking8[r].append(thinking8_i)
            all_answer8[r].append(answer8_i)
    final_decision_agent_8 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on the gamma-ray threshold energy choice.", is_sub_task=True)
    agents.append(f'Final Decision Agent {final_decision_agent_8.id}, selecting final answer, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer