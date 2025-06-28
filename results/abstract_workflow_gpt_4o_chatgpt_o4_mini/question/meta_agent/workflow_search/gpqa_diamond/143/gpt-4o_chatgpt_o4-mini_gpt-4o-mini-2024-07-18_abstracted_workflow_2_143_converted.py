async def forward_143(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Calculate the lifetime of the meson resonance
    cot_instruction_0 = "Sub-task 1: Calculate the lifetime of the meson resonance using the width (Γ_X = 320 MeV). The lifetime τ is given by τ = ħ/Γ, where ħ is the reduced Plancks constant (ħ ≈ 6.582 × 10^-22 MeV·s)."
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, calculating lifetime, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Calculate the velocity of the meson
    cot_sc_instruction_1 = "Sub-task 2: Calculate the velocity of the meson using its mass (m_X = 1.2 GeV) and production energy (E_X = 8 GeV). Use the relativistic relation v = pc/E, where p is the momentum and E is the energy. First, find the momentum using p = sqrt(E^2 - m^2)."
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents_1[i]([taskInfo, thinking1, answer1], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, calculating velocity, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_1.append(answer2.content)
        thinkingmapping_1[answer2.content] = thinking2
        answermapping_1[answer2.content] = answer2
    
    answer2 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking2 = thinkingmapping_1[answer2]
    answer2 = answermapping_1[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Calculate the mean decay distance
    cot_reflect_instruction_2 = "Sub-task 3: Calculate the mean decay distance using the formula d = vτ, where v is the velocity from subtask 2 and τ is the lifetime from subtask 1."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    
    cot_inputs_2 = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, calculating mean decay distance, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking3, answer3], 
                                       "Review mean decay distance calculation and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs_2.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining mean decay distance, thinking: {thinking3.content}; answer: {answer3.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 3: Compare the calculated mean decay distance with the given choices
    debate_instruction_3 = "Sub-task 4: Compare the calculated mean decay distance with the given choices to determine the correct answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3 = self.max_round
    
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], 
                                           debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking3, answer3] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking4, answer4 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, comparing mean decay distance, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_3[r].append(thinking4)
            all_answer_3[r].append(answer4)
    
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], 
                                                 "Sub-task 4: Make final decision on the correct mean decay distance.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, determining correct mean decay distance, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer