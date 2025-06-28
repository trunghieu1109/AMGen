async def forward_110(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_debate_instruction_1a = (
        "Sub-task 1a: Perform detailed conformational analysis of 2-ethyl-2,6-dimethylcyclohexan-1-one, "
        "including chair conformer identification, assignment of axial/equatorial positions for substituents, "
        "and evaluation of 1,3-diaxial interactions to determine the most stable conformer prior to reaction. "
        "Multiple agents will debate different conformers and stereochemical assignments to select the most plausible."
    )
    debate_agents_1a = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1a = self.max_round
    all_thinking_1a = [[] for _ in range(N_max_1a)]
    all_answer_1a = [[] for _ in range(N_max_1a)]
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_debate_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1a):
        for i, agent in enumerate(debate_agents_1a):
            if r == 0:
                thinking_1a, answer_1a = await agent([taskInfo], cot_debate_instruction_1a, r, is_sub_task=True)
            else:
                input_infos_1a = [taskInfo] + all_thinking_1a[r-1] + all_answer_1a[r-1]
                thinking_1a, answer_1a = await agent(input_infos_1a, cot_debate_instruction_1a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, conformational analysis, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
            all_thinking_1a[r].append(thinking_1a)
            all_answer_1a[r].append(answer_1a)
    final_decision_agent_1a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1a, answer_1a = await final_decision_agent_1a([taskInfo] + all_thinking_1a[-1] + all_answer_1a[-1], "Sub-task 1a: Select the most stable conformer and stereochemical assignment.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting conformer, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_sc_instruction_1b = (
        "Sub-task 1b: Analyze the enolate formation from 2-ethyl-2,6-dimethylcyclohexan-1-one under t-BuOK conditions, "
        "specifying the site of deprotonation, resonance stabilization, and stereochemical implications based on the conformer from subtask_1a. "
        "Use Self-Consistency Chain-of-Thought to generate multiple plausible mechanistic pathways and select the most consistent."
    )
    N = self.max_sc
    cot_agents_1b = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N)
    ]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_1b, answer_1b = await cot_agents_1b[i]([taskInfo, thinking_1a, answer_1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, enolate formation analysis, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
        possible_answers_1b.append(answer_1b.content)
        thinkingmapping_1b[answer_1b.content] = thinking_1b
        answermapping_1b[answer_1b.content] = answer_1b
    most_common_answer_1b = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking_1b = thinkingmapping_1b[most_common_answer_1b]
    answer_1b = answermapping_1b[most_common_answer_1b]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_sc_instruction_1c = (
        "Sub-task 1c: Analyze the Michael addition mechanism between the enolate intermediate and ethyl acrylate, "
        "including regiochemistry, stereochemistry of addition, and possible chair conformations of the product intermediate. "
        "Use Self-Consistency Chain-of-Thought to generate multiple plausible addition pathways and select the most consistent."
    )
    cot_agents_1c = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N)
    ]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_1c, answer_1c = await cot_agents_1c[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, Michael addition analysis, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
        possible_answers_1c.append(answer_1c.content)
        thinkingmapping_1c[answer_1c.content] = thinking_1c
        answermapping_1c[answer_1c.content] = answer_1c
    most_common_answer_1c = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking_1c = thinkingmapping_1c[most_common_answer_1c]
    answer_1c = answermapping_1c[most_common_answer_1c]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_instruction_1d = (
        "Sub-task 1d: Verify the carbon skeleton and substitution pattern of the predicted product A from the Michael addition, "
        "ensuring correct carbon count, ring size, and substituent positions consistent with stereochemical analysis. "
        "Use Chain-of-Thought with detailed structural mapping and cross-validation."
    )
    cot_agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_instruction_1d,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "CoT"
    }
    thinking_1d, answer_1d = await cot_agent_1d([taskInfo, thinking_1c, answer_1c], cot_instruction_1d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1d.id}, verifying product A structure, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_desc_1d['response'] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])
    
    cot_debate_instruction_2a = (
        "Sub-task 2a: Perform detailed conformational and tautomeric analysis of 1-nitropropane under KOH and aqueous conditions, "
        "including nitronate ion formation, resonance structures, and stereochemical considerations. "
        "Multiple agents will debate different resonance and tautomeric forms to select the most plausible."
    )
    debate_agents_2a = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_2a = self.max_round
    all_thinking_2a = [[] for _ in range(N_max_2a)]
    all_answer_2a = [[] for _ in range(N_max_2a)]
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_debate_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            if r == 0:
                thinking_2a, answer_2a = await agent([taskInfo], cot_debate_instruction_2a, r, is_sub_task=True)
            else:
                input_infos_2a = [taskInfo] + all_thinking_2a[r-1] + all_answer_2a[r-1]
                thinking_2a, answer_2a = await agent(input_infos_2a, cot_debate_instruction_2a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, nitronate formation analysis, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
            all_thinking_2a[r].append(thinking_2a)
            all_answer_2a[r].append(answer_2a)
    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2a, answer_2a = await final_decision_agent_2a([taskInfo] + all_thinking_2a[-1] + all_answer_2a[-1], "Sub-task 2a: Select the most plausible nitronate and tautomeric form.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting nitronate form, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_sc_instruction_2b = (
        "Sub-task 2b: Analyze the Michael addition between the nitronate ion and (E)-but-2-enenitrile, "
        "considering all possible addition sites, regiochemistry, stereochemistry, and tautomeric equilibria under the reaction conditions. "
        "Use Self-Consistency Chain-of-Thought to generate multiple plausible addition pathways and select the most consistent."
    )
    cot_agents_2b = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N)
    ]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, Michael addition analysis, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    most_common_answer_2b = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[most_common_answer_2b]
    answer_2b = answermapping_2b[most_common_answer_2b]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_2c = (
        "Sub-task 2c: Map the carbon skeleton and substituent positions of the predicted product B, "
        "verifying carbon count, regiochemistry, and stereochemical assignments to ensure structural correctness. "
        "Use Chain-of-Thought with detailed structural mapping and cross-validation."
    )
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "CoT"
    }
    thinking_2c, answer_2c = await cot_agent_2c([taskInfo, thinking_2b, answer_2b], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, verifying product B structure, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_sc_instruction_3a = (
        "Sub-task 3a: Generate multiple plausible product structures for reaction A based on outputs from subtask_1d, "
        "including alternative stereochemical and regiochemical isomers. Use Self-Consistency Chain-of-Thought."
    )
    cot_agents_3a = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N)
    ]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_3a, answer_3a = await cot_agents_3a[i]([taskInfo, thinking_1d, answer_1d], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, generating plausible products A, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
        possible_answers_3a.append(answer_3a.content)
        thinkingmapping_3a[answer_3a.content] = thinking_3a
        answermapping_3a[answer_3a.content] = answer_3a
    most_common_answer_3a = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking_3a = thinkingmapping_3a[most_common_answer_3a]
    answer_3a = answermapping_3a[most_common_answer_3a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_sc_instruction_3b = (
        "Sub-task 3b: Generate multiple plausible product structures for reaction B based on outputs from subtask_2c, "
        "including alternative stereochemical and regiochemical isomers. Use Self-Consistency Chain-of-Thought."
    )
    cot_agents_3b = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N)
    ]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_3b, answer_3b = await cot_agents_3b[i]([taskInfo, thinking_2c, answer_2c], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, generating plausible products B, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
        possible_answers_3b.append(answer_3b.content)
        thinkingmapping_3b[answer_3b.content] = thinking_3b
        answermapping_3b[answer_3b.content] = answer_3b
    most_common_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking_3b = thinkingmapping_3b[most_common_answer_3b]
    answer_3b = answermapping_3b[most_common_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = (
        "Sub-task 4: Evaluate all plausible products from subtasks 3a and 3b by analyzing steric hindrance, conformational stability, "
        "electronic effects, and solvent/kinetic influences to identify the most stable and major products for each reaction. "
        "Use Reflexion pattern with iterative critique and refinement."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking_3a, answer_3a, thinking_3b, answer_3b]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask_3a", "answer of subtask_3a", "thinking of subtask_3b", "answer of subtask_3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, evaluating product stability, thinking: {thinking_4.content}; answer: {answer_4.content}")
    for i in range(N_max_4):
        feedback_4, correct_4 = await critic_agent_4([taskInfo, thinking_4, answer_4], "please review the evaluation of steric hindrance, stability, and mechanistic consistency and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback_4.content}; answer: {correct_4.content}")
        if correct_4.content == "True":
            break
        cot_inputs_4.extend([thinking_4, answer_4, feedback_4])
        thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining evaluation, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_5 = (
        "Sub-task 5: Perform reflexive validation of the selected major products by cross-checking carbon counts, stereochemistry, regiochemistry, "
        "and mechanistic consistency; if inconsistencies are found, iterate back to relevant subtasks for correction."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking_4, answer_4]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Reflexion"
    }
    thinking_5, answer_5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, validating products, thinking: {thinking_5.content}; answer: {answer_5.content}")
    for i in range(N_max_5):
        feedback_5, correct_5 = await critic_agent_5([taskInfo, thinking_5, answer_5], "please review the product validation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback_5.content}; answer: {correct_5.content}")
        if correct_5.content == "True":
            break
        cot_inputs_5.extend([thinking_5, answer_5, feedback_5])
        thinking_5, answer_5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining validation, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_6 = (
        "Sub-task 6: Match the validated major products from subtask_5 with the given multiple-choice options by comparing detailed structural features and nomenclature "
        "to select the correct answer choice. Multiple agents will debate the matching to ensure accuracy."
    )
    debate_agents_6 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_6 = self.max_round
    all_thinking_6 = [[] for _ in range(N_max_6)]
    all_answer_6 = [[] for _ in range(N_max_6)]
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking_6, answer_6 = await agent([taskInfo, thinking_5, answer_5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking_5, answer_5] + all_thinking_6[r-1] + all_answer_6[r-1]
                thinking_6, answer_6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching products to choices, thinking: {thinking_6.content}; answer: {answer_6.content}")
            all_thinking_6[r].append(thinking_6)
            all_answer_6[r].append(answer_6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_6, answer_6 = await final_decision_agent_6([taskInfo] + all_thinking_6[-1] + all_answer_6[-1], "Sub-task 6: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct answer, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
