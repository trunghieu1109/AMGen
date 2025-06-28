async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Identify scalar fields and their VEVs, then rotate to mass eigenstates
    
    # Sub-task 1a: Identify all scalar fields and their VEVs
    cot_instruction_1a = (
        "Sub-task 1a: Identify all scalar fields in the model, including the singlet scalar φ, the Standard Model Higgs doublet H, "
        "and the scalar doublet S; determine which fields acquire vacuum expectation values (VEVs) and specify their values, "
        "clarifying that ⟨φ⟩ = x and ⟨h⟩ = v, while S does not acquire a VEV. Provide detailed justification."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, identifying scalar fields and VEVs, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    # Sub-task 1b: Rotate gauge eigenstates to mass eigenstates
    cot_instruction_1b = (
        "Sub-task 1b: Perform the rotation from gauge eigenstates to mass eigenstates for the scalar sector, explicitly defining the physical scalar states h₁, h₂ (with h₂ identified as the pseudo-Goldstone boson), "
        "charged scalars H^±, and pseudoscalar A⁰, and clarify their relation to the original fields and VEVs, based on output of Sub-task 1a."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, rotating to mass eigenstates, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    # Stage 2: Present theoretical framework for radiative mass generation
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Present the theoretical framework for radiative mass generation of pseudo-Goldstone bosons via the Coleman-Weinberg mechanism, "
        "including the explicit one-loop effective potential formula: V_eff = Σ (n_i / 64π²) M_i^4(ϕ) [ln(M_i²(ϕ)/μ²) - c_i], and explain how each particle’s loop contributes to the mass correction with corresponding coefficients α_i, "
        "based on output of Sub-task 1b."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1b, answer1b], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, presenting Coleman-Weinberg framework, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])
    
    # Stage 3: Enumerate and justify particle contributions and verify formula structure
    
    # Sub-task 3a: Scalar and gauge boson contributions
    cot_reflect_instruction_3a = (
        "Sub-task 3a: Enumerate and justify the scalar and gauge boson contributions to the one-loop radiative mass correction of the pseudo-Goldstone boson h₂, "
        "including h₁, W, Z, charged scalars H^±, neutral scalars H⁰, and pseudoscalar A⁰, specifying their roles and signs in the formula, based on output of Sub-task 2."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking2, answer2]
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, enumerating scalar and gauge boson contributions, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max_3a):
        feedback3a, correct3a = await critic_agent_3a([taskInfo, thinking3a, answer3a], "please review the scalar and gauge boson contributions and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback, thinking: {feedback3a.content}; answer: {correct3a.content}")
        if correct3a.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback3a])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining scalar and gauge boson contributions, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])
    
    # Sub-task 3b: Fermion contributions
    cot_reflect_instruction_3b = (
        "Sub-task 3b: Enumerate and justify the fermion contributions to the one-loop radiative mass correction, explicitly including the top quark and singlet fermions N_i, "
        "explaining the origin of their negative signs and their relative importance in the mass formula, based on output of Sub-task 2."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking2, answer2]
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, enumerating fermion contributions, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback3b, correct3b = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the fermion contributions and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback3b.content}; answer: {correct3b.content}")
        if correct3b.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback3b])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining fermion contributions, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])
    
    # Sub-task 3c: Verify VEV factor placement and formula structure
    cot_reflect_instruction_3c = (
        "Sub-task 3c: Analyze and confirm the correct placement of the VEV factor (x² + v²) in the denominator of the mass formula, "
        "and verify the overall structure and sign conventions of the radiative mass correction expression for h₂, based on outputs of Sub-tasks 3a and 3b."
    )
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking3a, answer3a, thinking3b, answer3b]
    subtask_desc_3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask_3a", "answer of subtask_3a", "thinking of subtask_3b", "answer of subtask_3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, verifying VEV factor and formula structure, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback3c, correct3c = await critic_agent_3c([taskInfo, thinking3c, answer3c], "please review the VEV factor placement and formula structure and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback3c.content}; answer: {correct3c.content}")
        if correct3c.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback3c])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining VEV factor and formula structure, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc_3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc_3c)
    print("Step 3c: ", sub_tasks[-1])
    
    # Stage 4: Construct approximate formula for squared mass M_h2^2
    debate_instruction_4 = (
        "Sub-task 4: Construct the approximate formula for the squared mass M_h₂² of the pseudo-Goldstone boson h₂ from radiative corrections, "
        "combining all particle contributions with their coefficients α_i and mass terms M_i⁴, ensuring correct dependence on (x² + v²) in the denominator and proper signs as justified in previous subtasks, "
        "based on outputs of Sub-task 3c."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask_3c", "answer of subtask_3c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3c, answer3c], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3c, answer3c] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, constructing formula for h2 mass, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the approximate formula for the squared mass M_h2².", is_sub_task=True)
    agents.append(f"Final Decision agent on formula construction, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Stage 5: Compare constructed formula with multiple-choice options
    debate_instruction_5 = (
        "Sub-task 5: Compare the constructed radiative mass formula with the provided multiple-choice options, critically evaluating each choice’s inclusion of particles, sign conventions, "
        "and VEV factor placement to select the correct approximation for M_h₂², based on output of Sub-task 4."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing formula with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct multiple-choice formula for the radiative mass correction of h2.", is_sub_task=True)
    agents.append(f"Final Decision agent on formula comparison, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
