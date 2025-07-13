async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Define variables and express numbers and sums (CoT collaboration)

    # Subtask 1: Define variables with domain constraints
    cot_instruction_1 = (
        "Sub-task 1: Define variables a11, a12, a13 for the first row and a21, a22, a23 for the second row, "
        "each an integer between 0 and 9 inclusive. Explicitly state that a11 >= 1 and a21 >= 1 to ensure the hundreds digits are nonzero, "
        "making the row numbers genuine three-digit numbers."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining variables and domain constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Express the two row-formed numbers as three-digit numbers
    cot_instruction_2 = (
        "Sub-task 2: Formally express the two row numbers as 100*a11 + 10*a12 + a13 and 100*a21 + 10*a22 + a23, "
        "reiterating that a11 >= 1 and a21 >= 1, and that leading zeros are allowed only in tens and ones places."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, expressing row numbers, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Express the three column-formed numbers as two-digit numbers
    cot_instruction_3 = (
        "Sub-task 3: Formally express the three column numbers as 10*a11 + a21, 10*a12 + a22, and 10*a13 + a23, "
        "stating that leading zeros are allowed in the tens place of these two-digit numbers, and digits are 0-9."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, expressing column numbers, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Write down the sum constraints explicitly
    cot_instruction_4 = (
        "Sub-task 4: Write the sum constraints: (100*a11 + 10*a12 + a13) + (100*a21 + 10*a22 + a23) = 999, "
        "and (10*a11 + a21) + (10*a12 + a22) + (10*a13 + a23) = 99, expressed algebraically in terms of the digit variables."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "stage_1_subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, writing sum constraints, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Algebraic deductions and system analysis (Debate and SC_CoT collaboration)

    # Subtask 1: Derive algebraic equation for row sum with Debate
    debate_instruction_21 = (
        "Sub-task 1: Derive and simplify the algebraic equation for the sum of the two row numbers in terms of the digit variables, "
        "considering place values and possible carryovers, and identify immediate digit constraints from sum=999. "
        "Avoid assuming individual digit pair sums equal 9. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_21 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_21 = self.max_round
    all_thinking_21 = [[] for _ in range(N_max_21)]
    all_answer_21 = [[] for _ in range(N_max_21)]
    subtask_desc21 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instruction_21,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_21):
        for i, agent in enumerate(debate_agents_21):
            if r == 0:
                thinking21, answer21 = await agent([taskInfo, thinking4], debate_instruction_21, r, is_sub_task=True)
            else:
                input_infos_21 = [taskInfo, thinking4] + all_thinking_21[r-1]
                thinking21, answer21 = await agent(input_infos_21, debate_instruction_21, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing row sum equation, thinking: {thinking21.content}; answer: {answer21.content}")
            all_thinking_21[r].append(thinking21)
            all_answer_21[r].append(answer21)
    final_decision_agent_21 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking21, answer21 = await final_decision_agent_21([taskInfo] + all_thinking_21[-1], "Sub-task 1: Synthesize and finalize the algebraic equation for row sum with constraints. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing row sum equation, thinking: {thinking21.content}; answer: {answer21.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking21.content}; answer - {answer21.content}")
    subtask_desc21['response'] = {"thinking": thinking21, "answer": answer21}
    logs.append(subtask_desc21)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 2: Derive algebraic equation for column sum with Debate
    debate_instruction_22 = (
        "Sub-task 2: Derive and simplify the algebraic equation for the sum of the three column numbers in terms of the digit variables, "
        "considering place values and possible carryovers, and identify digit constraints from sum=99. Avoid assumptions about individual digit sums. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_22 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_22 = self.max_round
    all_thinking_22 = [[] for _ in range(N_max_22)]
    all_answer_22 = [[] for _ in range(N_max_22)]
    subtask_desc22 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": debate_instruction_22,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_22):
        for i, agent in enumerate(debate_agents_22):
            if r == 0:
                thinking22, answer22 = await agent([taskInfo, thinking4], debate_instruction_22, r, is_sub_task=True)
            else:
                input_infos_22 = [taskInfo, thinking4] + all_thinking_22[r-1]
                thinking22, answer22 = await agent(input_infos_22, debate_instruction_22, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing column sum equation, thinking: {thinking22.content}; answer: {answer22.content}")
            all_thinking_22[r].append(thinking22)
            all_answer_22[r].append(answer22)
    final_decision_agent_22 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking22, answer22 = await final_decision_agent_22([taskInfo] + all_thinking_22[-1], "Sub-task 2: Synthesize and finalize the algebraic equation for column sum with constraints. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing column sum equation, thinking: {thinking22.content}; answer: {answer22.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking22.content}; answer - {answer22.content}")
    subtask_desc22['response'] = {"thinking": thinking22, "answer": answer22}
    logs.append(subtask_desc22)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 3: Combine the two sum equations into a system and analyze with SC_CoT
    cot_sc_instruction_23 = (
        "Sub-task 3: Combine the algebraic equations from Subtasks 1 and 2 into a system of linear equations relating the six digit variables. "
        "Analyze the system holistically to deduce relationships, dependencies, and bounds on digits, including carryover effects. "
        "Avoid assuming each digit pair sums to 9. Derive inequalities and possible digit ranges consistent with the system and domain constraints."
    )
    N_sc_23 = self.max_sc
    cot_agents_23 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_23)]
    possible_answers_23 = []
    possible_thinkings_23 = []
    subtask_desc23 = {
        "subtask_id": "stage_2_subtask_3",
        "instruction": cot_sc_instruction_23,
        "context": ["user query", thinking21.content, thinking22.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_23):
        thinking23, answer23 = await cot_agents_23[i]([taskInfo, thinking21, thinking22], cot_sc_instruction_23, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_23[i].id}, analyzing combined system, thinking: {thinking23.content}; answer: {answer23.content}")
        possible_answers_23.append(answer23)
        possible_thinkings_23.append(thinking23)
    final_decision_agent_23 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking23, answer23 = await final_decision_agent_23([taskInfo] + possible_thinkings_23, "Sub-task 3: Synthesize and finalize combined system analysis. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing combined system analysis, thinking: {thinking23.content}; answer: {answer23.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking23.content}; answer - {answer23.content}")
    subtask_desc23['response'] = {"thinking": thinking23, "answer": answer23}
    logs.append(subtask_desc23)
    print("Step 7: ", sub_tasks[-1])

    # Subtask 4: Identify and state all domain constraints explicitly (CoT)
    cot_instruction_24 = (
        "Sub-task 4: Explicitly state all domain constraints for each digit variable: digits are integers 0-9, "
        "and a11, a21 >= 1. Incorporate these constraints into the system and emphasize enforcement in enumeration and verification."
    )
    cot_agent_24 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc24 = {
        "subtask_id": "stage_2_subtask_4",
        "instruction": cot_instruction_24,
        "context": ["user query", thinking23.content],
        "agent_collaboration": "CoT"
    }
    thinking24, answer24 = await cot_agent_24([taskInfo, thinking23], cot_instruction_24, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_24.id}, stating domain constraints, thinking: {thinking24.content}; answer: {answer24.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking24.content}; answer - {answer24.content}")
    subtask_desc24['response'] = {"thinking": thinking24, "answer": answer24}
    logs.append(subtask_desc24)
    print("Step 8: ", sub_tasks[-1])

    # Subtask 5: Deduce additional logical constraints to reduce search space (SC_CoT)
    cot_sc_instruction_25 = (
        "Sub-task 5: Analyze the combined system and domain constraints to deduce additional logical constraints, "
        "such as possible carry values, digit sum bounds, or digit dependencies that reduce the search space. "
        "Document deductions clearly to guide enumeration, avoiding premature assumptions and verifying rigorously."
    )
    N_sc_25 = self.max_sc
    cot_agents_25 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_25)]
    possible_answers_25 = []
    possible_thinkings_25 = []
    subtask_desc25 = {
        "subtask_id": "stage_2_subtask_5",
        "instruction": cot_sc_instruction_25,
        "context": ["user query", thinking24.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_25):
        thinking25, answer25 = await cot_agents_25[i]([taskInfo, thinking24], cot_sc_instruction_25, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_25[i].id}, deducing logical constraints, thinking: {thinking25.content}; answer: {answer25.content}")
        possible_answers_25.append(answer25)
        possible_thinkings_25.append(thinking25)
    final_decision_agent_25 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking25, answer25 = await final_decision_agent_25([taskInfo] + possible_thinkings_25, "Sub-task 5: Synthesize and finalize logical constraints. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing logical constraints, thinking: {thinking25.content}; answer: {answer25.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking25.content}; answer - {answer25.content}")
    subtask_desc25['response'] = {"thinking": thinking25, "answer": answer25}
    logs.append(subtask_desc25)
    print("Step 9: ", sub_tasks[-1])

    # Stage 3: Enumeration and verification (SC_CoT and CoT collaboration)

    # Subtask 1: Enumerate all possible digit pair sums consistent with constraints (SC_CoT)
    cot_sc_instruction_31 = (
        "Sub-task 1: Enumerate all possible sums of digit pairs (a1j + a2j for j=1,2,3) consistent with the combined system and domain constraints, "
        "including carryover considerations. Enumerate at the level of digit sums, not individual digits, to reduce complexity. Avoid fixed sums like 9. "
        "Document all valid sum triples satisfying equations and constraints."
    )
    N_sc_31 = self.max_sc
    cot_agents_31 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_31)]
    possible_answers_31 = []
    possible_thinkings_31 = []
    subtask_desc31 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_sc_instruction_31,
        "context": ["user query", thinking25.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_31):
        thinking31, answer31 = await cot_agents_31[i]([taskInfo, thinking25], cot_sc_instruction_31, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_31[i].id}, enumerating digit pair sums, thinking: {thinking31.content}; answer: {answer31.content}")
        possible_answers_31.append(answer31)
        possible_thinkings_31.append(thinking31)
    final_decision_agent_31 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking31, answer31 = await final_decision_agent_31([taskInfo] + possible_thinkings_31, "Sub-task 1: Synthesize and finalize enumeration of digit pair sums. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing digit pair sums enumeration, thinking: {thinking31.content}; answer: {answer31.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking31.content}; answer - {answer31.content}")
    subtask_desc31['response'] = {"thinking": thinking31, "answer": answer31}
    logs.append(subtask_desc31)
    print("Step 10: ", sub_tasks[-1])

    # Subtask 2: For each valid digit sum triple, enumerate all digit assignments respecting domain constraints (SC_CoT)
    cot_sc_instruction_32 = (
        "Sub-task 2: For each valid digit sum triple from Sub-task 1, enumerate all possible digit assignments (a11,a21), (a12,a22), (a13,a23) satisfying sum constraints and domain restrictions, "
        "including a11 >= 1 and a21 >= 1. Filter out assignments violating digit bounds or nonzero hundreds digit rule. Break enumeration into manageable parts ensuring domain consistency."
    )
    N_sc_32 = self.max_sc
    cot_agents_32 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_32)]
    possible_answers_32 = []
    possible_thinkings_32 = []
    subtask_desc32 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_sc_instruction_32,
        "context": ["user query", thinking31.content, thinking24.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_32):
        thinking32, answer32 = await cot_agents_32[i]([taskInfo, thinking31, thinking24], cot_sc_instruction_32, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_32[i].id}, enumerating digit assignments, thinking: {thinking32.content}; answer: {answer32.content}")
        possible_answers_32.append(answer32)
        possible_thinkings_32.append(thinking32)
    final_decision_agent_32 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking32, answer32 = await final_decision_agent_32([taskInfo] + possible_thinkings_32, "Sub-task 2: Synthesize and finalize enumeration of digit assignments. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing digit assignments enumeration, thinking: {thinking32.content}; answer: {answer32.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking32.content}; answer - {answer32.content}")
    subtask_desc32['response'] = {"thinking": thinking32, "answer": answer32}
    logs.append(subtask_desc32)
    print("Step 11: ", sub_tasks[-1])

    # Subtask 3: Verify each enumerated digit assignment satisfies all constraints exactly (CoT)
    cot_instruction_33 = (
        "Sub-task 3: Verify each enumerated digit assignment from Sub-task 2 rigorously satisfies both row sum and column sum constraints exactly, including place value calculations and carryovers. "
        "Discard any invalid assignments. This verification must be automated and rigorous to prevent invalid solutions."
    )
    cot_agent_33 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc33 = {
        "subtask_id": "stage_3_subtask_3",
        "instruction": cot_instruction_33,
        "context": ["user query", thinking32.content],
        "agent_collaboration": "CoT"
    }
    thinking33, answer33 = await cot_agent_33([taskInfo, thinking32], cot_instruction_33, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_33.id}, verifying digit assignments, thinking: {thinking33.content}; answer: {answer33.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking33.content}; answer - {answer33.content}")
    subtask_desc33['response'] = {"thinking": thinking33, "answer": answer33}
    logs.append(subtask_desc33)
    print("Step 12: ", sub_tasks[-1])

    # Subtask 4: Count total number of valid digit assignments and report final answer (CoT)
    cot_instruction_34 = (
        "Sub-task 4: Count the total number of valid digit assignments that satisfy all constraints and report this count as the final answer. "
        "Provide a clear summary of the counting process and confirm all domain and sum constraints have been respected."
    )
    cot_agent_34 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc34 = {
        "subtask_id": "stage_3_subtask_4",
        "instruction": cot_instruction_34,
        "context": ["user query", thinking33.content],
        "agent_collaboration": "CoT"
    }
    thinking34, answer34 = await cot_agent_34([taskInfo, thinking33], cot_instruction_34, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_34.id}, counting valid assignments, thinking: {thinking34.content}; answer: {answer34.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking34.content}; answer - {answer34.content}")
    subtask_desc34['response'] = {"thinking": thinking34, "answer": answer34}
    logs.append(subtask_desc34)
    print("Step 13: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking34, answer34, sub_tasks, agents)
    return final_answer, logs
