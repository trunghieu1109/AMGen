async def forward_11(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and list all given physical quantities and constants from the problem statement, including rest masses of Pi(+) and mu(+), and the initial condition that Pi(+) is stationary."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identify physical quantities, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Write down the conservation laws applicable to the decay process Pi(+) → mu(+) + neutrino, specifically conservation of energy and conservation of momentum, considering Pi(+) is initially at rest."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, write conservation laws, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Express the total energy and momentum of the decay products (mu(+) and neutrino) using relativistic energy-momentum relations, incorporating rest masses and kinetic energies, and explicitly state assumptions such as neutrino mass = 0."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, express energy and momentum relations, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the relativistic energy-momentum expressions and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining expressions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_4a = "Sub-task 4a: Derive the expression for the muon momentum (p_mu) using conservation of momentum and energy, starting from the initial rest state of Pi(+), ensuring dimensional consistency and explicitly excluding neutrino mass."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_reflect_instruction_4a,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2", "thinking and answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, derive muon momentum expression, thinking: {thinking4a.content}; answer: {answer4a.content}")
    for i in range(N_max_4a):
        feedback, correct = await critic_agent_4a([taskInfo, thinking4a, answer4a], "please review the derivation of muon momentum and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4a.extend([thinking4a, answer4a, feedback])
        thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining muon momentum derivation, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_reflect_instruction_4b = "Sub-task 4b: Derive the kinetic energy (KE_mu) of the muon step-by-step from the muon momentum and rest mass, using relativistic energy relations, showing all intermediate steps and verifying dimensional correctness."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4a, answer4a]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_reflect_instruction_4b,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2", "thinking and answer of subtask 3", "thinking and answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, derive muon kinetic energy expression, thinking: {thinking4b.content}; answer: {answer4b.content}")
    for i in range(N_max_4b):
        feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b], "please review the derivation of muon kinetic energy and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4b.extend([thinking4b, answer4b, feedback])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining muon kinetic energy derivation, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4c = "Sub-task 4c: Derive the kinetic energy (KE_nu) of the neutrino using conservation of energy, subtracting the muon total energy from the initial Pi(+) rest energy, explicitly noting neutrino rest mass is zero."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4c = self.max_round
    cot_inputs_4c = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4b, answer4b]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_reflect_instruction_4c,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2", "thinking and answer of subtask 3", "thinking and answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, derive neutrino kinetic energy expression, thinking: {thinking4c.content}; answer: {answer4c.content}")
    for i in range(N_max_4c):
        feedback, correct = await critic_agent_4c([taskInfo, thinking4c, answer4c], "please review the derivation of neutrino kinetic energy and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4c.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4c.extend([thinking4c, answer4c, feedback])
        thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, refining neutrino kinetic energy derivation, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    
    cot_sc_instruction_5a = "Sub-task 5a: Numerically compute the muon kinetic energy (KE_mu) using the derived formula and given rest masses, showing all intermediate arithmetic steps (e.g., squaring masses, subtraction, division), and verify units and dimensional consistency."
    N = self.max_sc
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_sc_instruction_5a,
        "context": ["user query", "thinking and answer of subtask 4b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking4b, answer4b], cot_sc_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, compute muon KE numerically, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_sc_instruction_5b = "Sub-task 5b: Numerically compute the neutrino kinetic energy (KE_nu) using the derived formula and given rest masses, showing all intermediate arithmetic steps and verifying energy conservation (sum of kinetic energies plus rest masses equals initial rest mass energy)."
    N = self.max_sc
    cot_agents_5b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_sc_instruction_5b,
        "context": ["user query", "thinking and answer of subtask 4c", "thinking and answer of subtask 5a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5b, answer5b = await cot_agents_5b[i]([taskInfo, thinking4c, answer4c, thinking5a, answer5a], cot_sc_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5b[i].id}, compute neutrino KE numerically, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    answer5b_content = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[answer5b_content]
    answer5b = answermapping_5b[answer5b_content]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_reflect_instruction_5c = "Sub-task 5c: Perform a self-consistency check to verify that the sum of the computed kinetic energies and rest mass energies equals the initial Pi(+) rest mass energy, ensuring energy conservation."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5c = self.max_round
    cot_inputs_5c = [taskInfo, thinking5a, answer5a, thinking5b, answer5b]
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_reflect_instruction_5c,
        "context": ["user query", "thinking and answer of subtask 5a", "thinking and answer of subtask 5b"],
        "agent_collaboration": "Reflexion"
    }
    thinking5c, answer5c = await cot_agent_5c(cot_inputs_5c, cot_reflect_instruction_5c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5c.id}, verify energy conservation, thinking: {thinking5c.content}; answer: {answer5c.content}")
    for i in range(N_max_5c):
        feedback, correct = await critic_agent_5c([taskInfo, thinking5c, answer5c], "please review the energy conservation check and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5c.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5c.extend([thinking5c, answer5c, feedback])
        thinking5c, answer5c = await cot_agent_5c(cot_inputs_5c, cot_reflect_instruction_5c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5c.id}, refining energy conservation check, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Compare the computed kinetic energies (KE_mu and KE_nu) with the given multiple-choice options, checking both values against each choice within an acceptable tolerance, and select the correct choice that matches both kinetic energies."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking and answer of subtask 5c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5c, answer5c], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5c, answer5c] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compare kinetic energies with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct multiple-choice answer matching both kinetic energies.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs