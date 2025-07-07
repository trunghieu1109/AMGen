async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the given Lagrangian and field content to identify all scalar and fermion fields, their gauge charges, and their vacuum expectation values (VEVs), explicitly confirming <phi> = x and <h> = v. Summarize the global U(1) charges relevant to phi to set the symmetry context."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Lagrangian and VEVs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Identify which terms in the Lagrangian explicitly break the global symmetry associated with phi. List only those fields and couplings that involve phi and thus contribute to the pseudo-Goldstone boson mass through radiative corrections, explicitly excluding fields (e.g., top quark) that do not break this symmetry. Include a brief reminder of the global U(1) charges of each field to guide the analysis."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying symmetry-breaking terms, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    cot_reflect_instruction_3a = "Sub-task 3a: Derive the one-loop radiative correction contribution to the pseudo-Goldstone boson mass squared M_H2^2 from scalar fields identified in Sub-task 2, expressing it in terms of their masses and coupling coefficients alpha_i."
    cot_reflect_instruction_3b = "Sub-task 3b: Derive the one-loop radiative correction contribution to M_H2^2 from gauge bosons identified in Sub-task 2, expressing it in terms of their masses and coupling coefficients alpha_i."
    cot_reflect_instruction_3c = "Sub-task 3c: Derive the one-loop radiative correction contribution to M_H2^2 from fermions identified in Sub-task 2, explicitly excluding the top quark if it does not break the phi symmetry, expressing it in terms of their masses and coupling coefficients alpha_i."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, deriving scalar contributions, thinking: {thinking3a.content}; answer: {answer3a.content}")
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    for i in range(self.max_round):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], "please review the scalar contribution derivation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, scalar contribution feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2, thinking3a, answer3a, feedback], cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining scalar contributions, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking2, answer2], cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, deriving gauge boson contributions, thinking: {thinking3b.content}; answer: {answer3b.content}")
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    for i in range(self.max_round):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the gauge boson contribution derivation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, gauge boson contribution feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking2, answer2, thinking3b, answer3b, feedback], cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining gauge boson contributions, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c([taskInfo, thinking2, answer2], cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, deriving fermion contributions, thinking: {thinking3c.content}; answer: {answer3c.content}")
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    for i in range(self.max_round):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "please review the fermion contribution derivation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, fermion contribution feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking3c, answer3c = await cot_agent_3c([taskInfo, thinking2, answer2, thinking3c, answer3c, feedback], cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining fermion contributions, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    debate_instruction_4 = "Sub-task 4: Critically compare the derived expression for M_H2^2 with the provided choices. For each choice, analyze the placement of the overall factor (whether divided by or multiplied by (x^2 + v^2)), and the inclusion or exclusion of specific particle mass terms (notably the top quark and A^0). Use a debate-style reasoning to argue for and against each term's presence based on symmetry breaking and radiative correction principles."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing derived expression with choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct approximation for M_H2^2.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct approximation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    cot_instruction_5 = "Sub-task 5: Perform a final verification check ensuring the chosen formula for M_H2^2 includes all and only the contributions consistent with the global symmetry breaking analysis, specifically confirming the inclusion of the A^0 term and exclusion of the top quark term, and correct placement of the overall factor."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, performing final verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs