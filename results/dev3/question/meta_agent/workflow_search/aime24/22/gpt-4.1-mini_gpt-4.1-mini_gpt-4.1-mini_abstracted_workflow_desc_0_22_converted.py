async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Analyze and formalize all problem constraints precisely: the list consists of positive integers, sum equals 30, unique mode is 9, and the median is a positive integer not in the list. "
        "Explicitly deduce that the list length n must be even (to allow median as average of two middle elements not in the list) and that 9's frequency k must be at least 2 to be a unique mode. "
        "Define variables n (even), k (frequency of 9), and characterize the median as the average of the two middle elements. Avoid assuming list length or frequencies without justification. "
        "Establish assumptions such as the list being sorted in non-decreasing order and all elements positive integers. This subtask sets the rigorous foundation for subsequent numeric analysis and pruning."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Perform early constraint propagation and numeric feasibility analysis to prune impossible (n, k) pairs before enumeration. "
        "For each candidate even n and k >= 2, compute S = 30 - 9k and m = n - k (number of non-9 elements). "
        "Calculate the minimal sum of m distinct positive integers (1 + 2 + ... + m) and verify if min_sum <= S. If not, discard (n, k) pair. "
        "Also, analyze inequalities ensuring 9 is the unique mode (frequency(9) > frequency(other elements)). "
        "Produce a list of all viable (n, k) pairs that satisfy these numeric bounds. Document all assumptions and proofs used in pruning. Avoid enumerating full lists here."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, pruning (n,k) pairs, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze median feasibility given the viable (n, k) pairs from Sub-task 2. "
        "For each candidate n, determine all possible pairs of middle elements whose average is a positive integer not in the list. "
        "Characterize constraints on these middle elements (they must be consecutive integers differing by 2 to yield an integer median not in the list). "
        "Use these constraints to further prune or refine candidate (n, k) pairs and possible median values. Avoid premature enumeration of full lists. "
        "This subtask bridges frequency feasibility with median conditions, ensuring only consistent median candidates proceed."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing median feasibility, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_4 = (
        "Sub-task 4: Conduct a debate stage where agents argue the feasibility of different candidate frequencies k and list lengths n based on the inequality and median feasibility results. "
        "Assign agents to advocate for specific (n, k) pairs or median candidates, while a critic agent adjudicates based on numeric proofs and constraints. "
        "The goal is to reach consensus on a minimal set of plausible (n, k, median) triples for enumeration. "
        "Explicitly document rejected candidates with reasons."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, arguing feasibility, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Finalize plausible (n,k,median) triples for enumeration." + " Given all debate outputs, provide a minimal set of valid triples.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing candidate triples, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Enumerate all candidate lists consistent with the refined (n, k, median) triples from Sub-task 4. "
        "For each candidate list, generate elements respecting: sum=30, frequency(9)=k, unique mode=9, median equals the specified integer not in the list, and all elements positive integers sorted non-decreasingly. "
        "Implement pruning using earlier numeric bounds to avoid brute-force enumeration. For each candidate list, verify all constraints strictly. "
        "If no valid list exists, explicitly output 'No valid list exists' to avoid fallback on invalid solutions."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, enumerating candidate lists, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 6: For each valid list from Sub-task 5, compute the sum of the squares of all items. "
        "Confirm that the computed sum corresponds to a list satisfying all constraints. If multiple valid lists exist, verify whether the sum of squares is unique or identify all possible sums. "
        "Provide the final answer as the sum of squares of the list(s) that meet all conditions. Avoid reporting answers from invalid or approximate lists."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, computing sum of squares, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    best_answer_6 = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[best_answer_6]
    answer6 = answermapping_6[best_answer_6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_7 = (
        "Sub-task 7: Perform a comprehensive verification and reflexion stage. Cross-check the final answer(s) from Sub-task 6 against all problem constraints: sum, unique mode, median condition, and sum of squares calculation. "
        "Reflect on the entire reasoning process, confirm no contradictions or overlooked cases remain, and synthesize verification feedback. "
        "If inconsistencies or multiple answers arise, provide a reasoned resolution or declare no unique solution. "
        "Output the final verified answer alongside verification results in strict JSON format."
    )
    cot_reflect_instruction_7 = "Sub-task 7: Verify and refine the final answer." + reflect_inst_7
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying and refining solution, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7],
                                               "Please review the answer above and criticize on where might be wrong. "
                                               "If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining solution, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
