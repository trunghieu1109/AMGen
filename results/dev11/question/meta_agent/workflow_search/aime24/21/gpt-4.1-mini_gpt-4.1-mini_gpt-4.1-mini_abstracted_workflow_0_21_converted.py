async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalization and Geometric Analysis

    # Sub-task 0.1: Formalize the problem setting (CoT | SC_CoT)
    cot_instruction_0_1 = (
        "Sub-task 0.1: Formalize the problem setting by rigorously defining the regular dodecagon, "
        "including vertices, sides, diagonals, and clarify the meaning of rectangle sides lying on sides or diagonals. "
        "Explicitly state geometric constraints for rectangles and assumptions such as convexity and boundary sharing. "
        "Avoid ambiguous definitions and unjustified assumptions."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing problem, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Sub-task 0.2: Analyze geometric and combinatorial structure of dodecagon chords (CoT | SC_CoT)
    cot_instruction_0_2 = (
        "Sub-task 0.2: Analyze the geometric and combinatorial structure of the dodecagon to characterize all sides and diagonals as chords. "
        "Identify chord properties including lengths and directions (skip-lengths), determine candidates for rectangle sides, "
        "examine parallel and perpendicular chord pairs, and identify symmetries simplifying the problem. "
        "Avoid premature enumeration; establish necessary and sufficient conditions for rectangle sides."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing chord structure, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Stage 1: Combinatorial Framework and Enumeration

    # Sub-task 1.1: Develop combinatorial framework for enumeration (CoT | SC_CoT | Reflexion)
    cot_sc_instruction_1_1 = (
        "Sub-task 1.1: Develop a combinatorial framework translating geometric conditions into algebraic and combinatorial criteria to enumerate rectangles. "
        "Classify chords by skip-length k (1 to 6), identify perpendicular pairs (k,l), establish rectangle formation criteria, "
        "and define systematic vertex quadruple selection. Use polygon symmetries to reduce complexity and avoid double counting. "
        "Avoid brute force enumeration; prune impossible or redundant configurations."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, combinatorial framework, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1.1: Synthesize and choose the most consistent combinatorial framework.", is_sub_task=True)
    agents.append(f"Final Decision agent, combinatorial framework, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Sub-task 1.2: Detailed enumeration of rectangles by chord skip-length pairs (CoT | SC_CoT | Reflexion | Debate)
    cot_sc_instruction_1_2 = (
        "Sub-task 1.2: Perform detailed enumeration of all rectangles formed by polygon sides and diagonals using the combinatorial framework. "
        "For each perpendicular pair (k,l) of skip-lengths, count rectangles formed, produce a comprehensive enumeration table with entries: "
        "side skip-length k | perpendicular skip-length l | number of rectangles. Ensure no duplicates or omissions by symmetry and verification. "
        "Account for rectangles sharing edges with polygon boundary and those strictly inside. Avoid accepting classical results without rigorous enumeration and cross-validation."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, detailed enumeration, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 1.2: Synthesize and choose the most consistent detailed enumeration.", is_sub_task=True)
    agents.append(f"Final Decision agent, detailed enumeration, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Sub-task 1.3: Sanity check by explicit listing of representative rectangles (CoT | Reflexion | Debate)
    cot_reflect_instruction_1_3 = (
        "Sub-task 1.3: Conduct sanity check by selecting representative entries from enumeration table (e.g., k=1, l=2). "
        "Explicitly list or illustrate vertex quadruples forming rectangles for these cases to verify counts. "
        "Identify and resolve discrepancies or ambiguities. Provide concrete verification to prevent flawed counts."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, sanity check, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review and provide limitations or confirm correctness. If correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback_1_3.content}; correct: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining sanity check, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Sub-task 1.4: Reconciliation analysis of conflicting counts (Debate | Reflexion)
    debate_instruction_1_4 = (
        "Sub-task 1.4: Perform reconciliation analysis of conflicting counts from enumeration and classical claims. "
        "Identify sources of overcounting or undercounting, use symmetry group actions to count distinct rectangles accurately, "
        "update enumeration table accordingly and justify corrections. Avoid premature acceptance of any count without reconciliation."
    )
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_1_4,
        "context": ["user query", thinking_1_2.content, answer_1_2.content, thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos_1_4 = [taskInfo, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3] + all_thinking_1_4[r-1] + all_answer_1_4[r-1]
                thinking_i, answer_i = await agent(input_infos_1_4, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, reconciliation, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_1_4[r].append(thinking_i)
            all_answer_1_4[r].append(answer_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + all_thinking_1_4[-1] + all_answer_1_4[-1], "Sub-task 1.4: Final reconciliation and corrected enumeration table.", is_sub_task=True)
    agents.append(f"Final Decision agent, reconciliation, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    # Stage 2: Final Computation and Verification

    # Sub-task 2.1: Compute final total number of rectangles and verify (CoT | Reflexion)
    cot_reflect_instruction_2_1 = (
        "Sub-task 2.1: Compute the final total number of rectangles by summing all entries in the reconciled enumeration table. "
        "Cross-check the final result with geometric properties, symmetry considerations, and known results if available. "
        "Reflect critically on completeness and correctness, explicitly stating any remaining assumptions or limitations. "
        "Provide the final answer with detailed verification report including enumeration data, sanity checks, and reconciliation outcomes. "
        "Avoid issuing final answer without thorough verification and documentation."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking_1_4, answer_1_4]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, final computation and verification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
