async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Formal Problem Definition
    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem variables and constraints for the 2x3 digit grid. "
        "Express the two three-digit numbers formed by rows as (100*a + 10*b + c) and (100*d + 10*e + f), "
        "and the three two-digit numbers formed by columns as (10*a + d), (10*b + e), and (10*c + f). "
        "Confirm that leading zeros are allowed, digits 0-9 may be freely assigned, and the sum constraints are: "
        "row1 + row2 = 999 and col1 + col2 + col3 = 99. Avoid assumptions about digit relationships or carries at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formal problem definition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Stage 1 Subtask 2: Enumerate carry scenarios for row addition
    cot_sc_instruction_2 = (
        "Sub-task 2: Enumerate all possible carry scenarios for the addition of the two three-digit row numbers summing to 999. "
        "Introduce explicit carry variables k1 and k2 (each in {0,1}) for units and tens digit additions respectively. "
        "Formulate digit-wise addition constraints with carries as: c + f = 9 + 10*k1, b + e + k1 = 9 + 10*k2, a + d + k2 = 9. "
        "Enumerate all valid (k1, k2) pairs and derive corresponding digit sum relations. Avoid collapsing constraints or assuming fixed sums without carries."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, enumerate carry scenarios, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct carry scenarios and digit sum relations."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize carry scenarios." + final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Stage 2 Subtask 1: Enumerate all digit triples (a,b,c) and (d,e,f) satisfying carry constraints
    cot_sc_instruction_3 = (
        "Sub-task 1: Enumerate all digit triples (a,b,c) and (d,e,f) with digits 0-9, without premature pruning. "
        "For each pair, verify digit-wise addition constraints derived from carry scenarios in stage_1.subtask_2. "
        "Check that c+f, b+e+k1, a+d+k2 meet conditions for each carry scenario. "
        "Explicitly test all pairs to avoid missing valid solutions."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, enumerate digit triples with carry constraints, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, synthesize all valid digit triple pairs satisfying carry constraints."
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 1: Synthesize digit triples." + final_instr_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Stage 2 Subtask 2: Apply column sum constraint to filter digit triples
    cot_sc_instruction_4 = (
        "Sub-task 2: Apply the column sum constraint ((10*a + d) + (10*b + e) + (10*c + f) = 99) to filter digit triples from stage_2.subtask_1. "
        "For each candidate, verify the sum of the three two-digit column numbers equals 99. "
        "Integrate this with carry-aware digit sum constraints to identify consistent assignments. "
        "Avoid treating row and column constraints independently without cross-verification."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, apply column sum constraint, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, synthesize all valid digit assignments satisfying both row and column constraints."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 2: Synthesize valid digit assignments." + final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    # Stage 3 Subtask 1: Algebraic simplification and modular arithmetic analysis
    debate_instruction_5 = (
        "Sub-task 1: Perform algebraic simplification and modular arithmetic analysis on combined constraints from stage_2.subtask_2. "
        "Reduce search space, identify necessary digit relationships, detect contradictions or forced digit values, and prune invalid candidates efficiently. "
        "Document assumptions and validate against enumeration results."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, algebraic simplification, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 1: Algebraic simplification and pruning." + final_instr_5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    # Stage 3 Subtask 2: Comprehensive verification of all constraints
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_6 = (
        "Sub-task 2: Perform comprehensive verification where all candidate digit assignments are cross-checked simultaneously against every constraint: digit-wise addition with carries, row sum = 999, column sum = 99, and digit validity (0-9). "
        "Confirm no contradictions and document discrepancies or edge cases. This step avoids undercounting or overcounting solutions. "
        + reflect_inst
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, comprehensive verification, thinking: {thinking6.content}; answer: {answer6.content}")
    critic_inst_6 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], critic_inst_6, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining verification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    # Stage 4 Subtask 1: Aggregate all verified valid digit assignments
    cot_instruction_7 = (
        "Sub-task 1: Aggregate all verified valid digit assignments from stage_3.subtask_2 to compute the total number of ways to fill the 2x3 grid satisfying both sum constraints. "
        "Provide the final count with detailed explanation of verification results and reasoning ensuring correctness. "
        "Avoid premature conclusions or ignoring any valid solutions."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, aggregate final count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
