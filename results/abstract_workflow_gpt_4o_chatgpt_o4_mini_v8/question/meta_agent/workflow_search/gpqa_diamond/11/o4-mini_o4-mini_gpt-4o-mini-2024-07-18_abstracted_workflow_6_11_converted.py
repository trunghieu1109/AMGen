async def forward_11(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction = 'Sub-task 1: Extract the rest mass of the pion m_pi = 139.6 MeV from taskInfo.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, extracting m_pi, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 2: Extract the rest mass of the muon m_mu = 105.7 MeV from taskInfo.'
    cot_agent2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent2.id}, extracting m_mu, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 3: State that initial momentum is zero and that momentum and energy are conserved in the decay pi+ -> mu+ + nu.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, conservation principles, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 4: Write momentum conservation p_mu + p_nu = 0 so |p_mu| = |p_nu| = p in the pion rest frame.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, momentum conservation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 5: Write energy conservation m_pi = E_mu + E_nu with E_mu and E_nu as total energies.'
    cot_agent5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent5.id}, energy conservation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 6: Express E_mu = sqrt(p^2 + m_mu^2) and E_nu = p.'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking2, answer2, thinking4, answer4, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, energy expressions, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])
    sc_instruction = 'Sub-task 7: Solve m_pi = sqrt(p^2 + m_mu^2) + p algebraically for p = (m_pi^2 - m_mu^2)/(2*m_pi).'
    N = self.max_sc
    cot_agents7 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    answers7 = []
    for i in range(N):
        thinking7, answer7 = await cot_agents7[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking6, answer6], sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents7[i].id}, solving for p, thinking: {thinking7.content}; answer: {answer7.content}')
        answers7.append(answer7.content)
    p_solution = Counter(answers7).most_common(1)[0][0]
    sub_tasks.append(f'Sub-task 7 output: p = {p_solution}')
    print('Step 7: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 8: Substitute m_pi = 139.6 MeV and m_mu = 105.7 MeV into p formula to compute p numerically in MeV/c.'
    cot_agent8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent8([taskInfo, p_solution], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent8.id}, numeric p calculation, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print('Step 8: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 9: Compute neutrino kinetic energy KE_nu = p.'
    cot_agent9 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent9([taskInfo, answer8], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent9.id}, KE_nu calculation, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    print('Step 9: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 10: Compute muon kinetic energy KE_mu = sqrt(p^2 + m_mu^2) - m_mu using p from sub-task 8 and m_mu = 105.7 MeV.'
    cot_agent10 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking10, answer10 = await cot_agent10([taskInfo, answer8, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent10.id}, KE_mu calculation, thinking: {thinking10.content}; answer: {answer10.content}')
    sub_tasks.append(f'Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}')
    print('Step 10: ', sub_tasks[-1])
    debate_instruction = 'Sub-task 11: Compare KE_nu and KE_mu to the provided choices and select the matching choice.'
    debate_agents11 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max11 = self.max_round
    all_thinking11 = [[] for _ in range(N_max11)]
    all_answer11 = [[] for _ in range(N_max11)]
    for r in range(N_max11):
        for i, agent in enumerate(debate_agents11):
            inputs = [taskInfo, answer9, answer10]
            if r > 0:
                inputs += all_thinking11[r-1] + all_answer11[r-1]
            thinking11, answer11 = await agent(inputs, debate_instruction, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, comparing choices, thinking: {thinking11.content}; answer: {answer11.content}')
            all_thinking11[r].append(thinking11)
            all_answer11[r].append(answer11)
    final_decision_agent11 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking11, answer11 = await final_decision_agent11([taskInfo] + all_thinking11[-1] + all_answer11[-1], 'Sub-task 11: Make final decision on choice.', is_sub_task=True)
    agents.append(f'Final Decision agent, final choice selection, thinking: {thinking11.content}; answer: {answer11.content}')
    sub_tasks.append(f'Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}')
    print('Step 11: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer