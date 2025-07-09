async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction0_1 = "Sub-task 0.1: Parse the full Lagrangian to extract all fields (N_R, S, φ, H), their quantum numbers, kinetic terms, and interaction terms under the SM gauge group and any global symmetries."
    cot_agent0_1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0_1 = {"subtask_id":"subtask_0_1","instruction":cot_instruction0_1,"context":["user query"],"agent_collaboration":"CoT"}
    thinking0_1, answer0_1 = await cot_agent0_1([taskInfo], cot_instruction0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent0_1.id}, parsing Lagrangian, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc0_1['response']={"thinking":thinking0_1,"answer":answer0_1}
    logs.append(subtask_desc0_1)
    cot_instruction0_2 = "Sub-task 0.2: Isolate from the Lagrangian the scalar potential V(φ,S,H) and all Yukawa couplings, including φ–H mixing terms and N_i–φ and N_i–L–S couplings, explicitly listing coupling constants."
    cot_agent0_2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0_2 = {"subtask_id":"subtask_0_2","instruction":cot_instruction0_2,"context":["user query","response subtask_0_1"],"agent_collaboration":"CoT"}
    thinking0_2, answer0_2 = await cot_agent0_2([taskInfo, thinking0_1, answer0_1], cot_instruction0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent0_2.id}, isolating potential and Yukawa terms, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc0_2['response']={"thinking":thinking0_2,"answer":answer0_2}
    logs.append(subtask_desc0_2)
    cot_instruction1_1 = "Sub-task 1.1: Identify the vacuum expectation values ⟨φ⟩=x and ⟨H⟩=v, record the combined symmetry-breaking scale (x²+v²), and note which terms in V induce spontaneous breaking."
    cot_agent1_1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1_1 = {"subtask_id":"subtask_1_1","instruction":cot_instruction1_1,"context":["user query","response subtask_0_2"],"agent_collaboration":"CoT"}
    thinking1_1, answer1_1 = await cot_agent1_1([taskInfo, thinking0_2, answer0_2], cot_instruction1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1_1.id}, identifying VEVs and breaking terms, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc1_1['response']={"thinking":thinking1_1,"answer":answer1_1}
    logs.append(subtask_desc1_1)
    cot_reflect_instruction1_2 = "Sub-task 1.2: Determine which global symmetry is broken by these VEVs, identify the Goldstone modes, and single out the pseudo-Goldstone boson H₂ whose tree-level mass vanishes in the exact symmetry limit."
    cot_agent1_2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent1_2 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max1_2 = self.max_round
    cot_inputs1_2 = [taskInfo, thinking1_1, answer1_1]
    subtask_desc1_2 = {"subtask_id":"subtask_1_2","instruction":cot_reflect_instruction1_2,"context":["user query","response subtask_1_1"],"agent_collaboration":"Reflexion"}
    thinking1_2, answer1_2 = await cot_agent1_2(cot_inputs1_2, cot_reflect_instruction1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent1_2.id}, identifying pseudo-Goldstone H2, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    for i in range(N_max1_2):
        feedback1_2, correct1_2 = await critic_agent1_2([taskInfo, thinking1_2, answer1_2], "Please review the identification of H2 and its tree-level mass limit.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent1_2.id}, feedback: {feedback1_2.content}; correct: {correct1_2.content}")
        if correct1_2.content == 'True':
            break
        cot_inputs1_2.extend([thinking1_2, answer1_2, feedback1_2])
        thinking1_2, answer1_2 = await cot_agent1_2(cot_inputs1_2, cot_reflect_instruction1_2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent1_2.id}, refining H2 identification, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc1_2['response']={"thinking":thinking1_2,"answer":answer1_2}
    logs.append(subtask_desc1_2)
    cot_instruction2_1 = "Sub-task 2.1: Enumerate all one-loop self-energy diagrams contributing to H₂, listing internal lines for gauge bosons (W,Z), scalars (h₁,H⁰,H^±,A⁰), heavy singlet fermions N_i, and SM fermions (especially the top quark)."
    cot_agent2_1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {"subtask_id":"subtask_2_1","instruction":cot_instruction2_1,"context":["user query","response subtask_1_2"],"agent_collaboration":"CoT"}
    thinking2_1, answer2_1 = await cot_agent2_1([taskInfo, thinking1_2, answer1_2], cot_instruction2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2_1.id}, listing one-loop diagrams, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc2_1['response']={"thinking":thinking2_1,"answer":answer2_1}
    logs.append(subtask_desc2_1)
    cot_instruction2_2 = "Sub-task 2.2: Write down the generic one-loop mass correction ΔM² = (1/16π²) Σ_i η_i M_i⁴/(x²+v²), assigning the correct sign (η_i=+ for bosons, – for fermions) and multiplicity factors to each loop."
    cot_agent2_2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_2 = {"subtask_id":"subtask_2_2","instruction":cot_instruction2_2,"context":["user query","response subtask_2_1"],"agent_collaboration":"CoT"}
    thinking2_2, answer2_2 = await cot_agent2_2([taskInfo, thinking2_1, answer2_1], cot_instruction2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2_2.id}, writing mass correction formula, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc2_2['response']={"thinking":thinking2_2,"answer":answer2_2}
    logs.append(subtask_desc2_2)
    cot_instruction2_3 = "Sub-task 2.3: Aggregate the contributions into the explicit form M_{h₂}² = 1/[8π²(x²+v²)]{α₁M_{h₁}⁴+α₂M_W⁴+α₃M_Z⁴+…–α_tM_t⁴–α_NΣM_{N_i}⁴}, ensuring inclusion of all scalars, gauge bosons, top-quark, and N_i terms with correct coefficients."
    cot_agent2_3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_3 = {"subtask_id":"subtask_2_3","instruction":cot_instruction2_3,"context":["user query","response subtask_2_2"],"agent_collaboration":"CoT"}
    thinking2_3, answer2_3 = await cot_agent2_3([taskInfo, thinking2_2, answer2_2], cot_instruction2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2_3.id}, aggregating contributions, thinking: {thinking2_3.content}; answer: {answer2_3.content}")
    sub_tasks.append(f"Sub-task 2_3 output: thinking - {thinking2_3.content}; answer - {answer2_3.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc2_3['response']={"thinking":thinking2_3,"answer":answer2_3}
    logs.append(subtask_desc2_3)
    cot_sc_instruction3_1 = "Sub-task 3.1: Compare the derived explicit M_{h₂}² expression term-by-term against the four provided answer choices, checking which particles appear, the sign of each term, and the prefactor."
    N1 = self.max_sc
    cot_agents3_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_answers3_1 = []
    thinking_map3_1 = {}
    answer_map3_1 = {}
    subtask_desc3_1 = {"subtask_id":"subtask_3_1","instruction":cot_sc_instruction3_1,"context":["user query","response subtask_2_3"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking_i3_1, answer_i3_1 = await cot_agents3_1[i]([taskInfo, thinking2_3, answer2_3], cot_sc_instruction3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3_1[i].id}, comparing expressions, thinking: {thinking_i3_1.content}; answer: {answer_i3_1.content}")
        possible_answers3_1.append(answer_i3_1.content)
        thinking_map3_1[answer_i3_1.content] = thinking_i3_1
        answer_map3_1[answer_i3_1.content] = answer_i3_1
    answer3_1_content = Counter(possible_answers3_1).most_common(1)[0][0]
    thinking3_1 = thinking_map3_1[answer3_1_content]
    answer3_1 = answer_map3_1[answer3_1_content]
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    print("Step 8: ", sub_tasks[-1])
    subtask_desc3_1['response']={"thinking":thinking3_1,"answer":answer3_1}
    logs.append(subtask_desc3_1)
    debate_instruction3_2 = "Sub-task 3.2: Using debate, select the letter (A, B, C, or D) whose structure of bosonic and fermionic loops, overall sign conventions, and prefactor exactly match the derived formula."
    debate_agents3_2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max3_2 = self.max_round
    all_thinking3_2 = [[] for _ in range(N_max3_2)]
    all_answer3_2 = [[] for _ in range(N_max3_2)]
    subtask_desc3_2 = {"subtask_id":"subtask_3_2","instruction":debate_instruction3_2,"context":["user query","response subtask_3_1"],"agent_collaboration":"Debate"}
    for r in range(N_max3_2):
        for i, agent in enumerate(debate_agents3_2):
            if r == 0:
                thinking3_2, answer3_2 = await agent([taskInfo, thinking3_1, answer3_1], debate_instruction3_2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3_1, answer3_1] + all_thinking3_2[r-1] + all_answer3_2[r-1]
                thinking3_2, answer3_2 = await agent(inputs, debate_instruction3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
            all_thinking3_2[r].append(thinking3_2)
            all_answer3_2[r].append(answer3_2)
    final_decision_agent3_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent3_2([taskInfo] + all_thinking3_2[-1] + all_answer3_2[-1], "Sub-task 3.2: Make final decision on the matching choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3_2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    print("Step 9: ", sub_tasks[-1])
    subtask_desc3_2['response']={"thinking":thinking3_2,"answer":answer3_2}
    logs.append(subtask_desc3_2)
    final_answer = await self.make_final_answer(thinking3_2, answer3_2, sub_tasks, agents)
    return final_answer, logs