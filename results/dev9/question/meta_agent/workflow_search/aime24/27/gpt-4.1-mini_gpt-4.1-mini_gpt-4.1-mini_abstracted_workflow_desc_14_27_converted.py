async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state all problem constraints and conditions on N. "
        "Confirm that N is a four-digit integer (1000 ≤ N ≤ 9999) with digits a, b, c, d (a ≥ 1, 0 ≤ b,c,d ≤ 9). "
        "Define precisely the operation of changing any one digit of N to 1, including the case when the digit is already 1 (the digit is replaced regardless). "
        "Enumerate the digit positions (thousands, hundreds, tens, units) and clarify that the resulting number after each single-digit replacement must be divisible by 7. "
        "Avoid assuming any digit restrictions beyond those stated. This subtask sets the foundation for all subsequent algebraic and modular reasoning."
    )
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, identifying problem constraints, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_answers_0_1 + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent answer for problem constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Express each modified number formed by changing exactly one digit of N to 1 in terms of the digits a, b, c, d. "
        "Formulate the divisibility condition for each modified number by 7, representing N as 1000a + 100b + 10c + d. "
        "For each digit position i, write the modified number as N with digit d_i replaced by 1. Avoid premature assumptions about digit values; focus on expressing these conditions algebraically and preparing for modular arithmetic formulation."
    )
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, expressing modified numbers and divisibility conditions, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent algebraic expressions for modified numbers.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Derive the modular arithmetic equations from the divisibility conditions obtained in Stage 0. "
        "For each digit position, write the congruence that the modified number is divisible by 7. "
        "Translate these into modular equations involving digits a, b, c, d modulo 7. "
        "Emphasize the relationship between the original digit and 1, and how the difference affects divisibility by 7. "
        "Avoid skipping steps in modular reasoning to ensure clarity and correctness. This subtask must produce a complete system of modular equations ready for algebraic solving."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_2, answer_0_2],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving modular equations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Algebraically solve the full system of modular equations simultaneously to determine the digits a, b, c modulo 7. "
        "Explicitly incorporate all modular constraints, including the previously overlooked third equation, to uniquely characterize the digits modulo 7 before any candidate enumeration. "
        "Implement symbolic checks to reject any candidate digit values that do not satisfy all modular equations. Avoid brute-force guessing or partial solving. This ensures correctness and completeness of the modular solution."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, algebraically solving modular system, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_2.append(answer)
        possible_thinkings_1_2.append(thinking)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent algebraic solution for digits.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Enumerate all possible digit quadruples (a, b, c, d) within digit bounds (a ≥ 1, 0 ≤ b,c,d ≤ 9) that satisfy the modular constraints derived and solved in Stage 1. "
        "For each candidate quadruple, explicitly compute all four modified numbers formed by changing each digit to 1. "
        "This enumeration must be systematic and exhaustive over the feasible digit ranges, leveraging modular results to prune invalid candidates early. "
        "Avoid premature acceptance of candidates without full modular compliance."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, enumerating candidate digits and modified numbers, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: For each candidate quadruple from subtask_1, perform explicit, step-by-step divisibility checks by 7 on all four modified numbers. "
        "Show detailed arithmetic calculations (e.g., division steps, remainders) to avoid errors. Only accept candidates passing all four divisibility tests. "
        "This subtask enforces rigorous validation of the problem's core condition and prevents acceptance of invalid solutions. Avoid skipping or approximating divisibility checks."
    )
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, performing divisibility checks, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent valid candidates after divisibility checks.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_2_3 = (
        "Sub-task 3: From the validated candidates, identify the greatest four-digit number N satisfying all conditions. "
        "Re-verify all divisibility conditions explicitly for this chosen N to ensure no oversight. "
        "This final verification step prevents premature conclusions and confirms the correctness of the solution before proceeding to final computations. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": debate_instruction_2_3,
        "context": ["user query", thinking_2_2, answer_2_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_2_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_2_3[r-1] + all_answer_2_3[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying and selecting greatest N, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_3[r].append(thinking)
            all_answer_2_3[r].append(answer)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking_2_3[-1] + all_answer_2_3[-1], "Sub-task 3: Final verification and selection of greatest N.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_2_4 = (
        "Sub-task 4: Compute Q and R from the chosen N, where Q is the quotient and R the remainder when dividing N by 1000 (i.e., Q = thousands digit, R = last three digits). "
        "Prepare these values for the final summation. Avoid errors in digit extraction or division."
    )
    cot_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_instruction_2_4,
        "context": ["user query", thinking_2_3, answer_2_3],
        "agent_collaboration": "CoT"
    }
    thinking_2_4, answer_2_4 = await cot_agent_2_4([taskInfo, thinking_2_3, answer_2_3], cot_instruction_2_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_4.id}, computing Q and R from N, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 8: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Compute and return the sum Q + R as the final answer. "
        "Provide a clear final answer along with a brief verification summary confirming that N and all its digit-to-1 modifications are divisible by 7. "
        "Ensure no ambiguity in the final output and that the answer aligns with the problem requirements. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_4, answer_2_4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_4, answer_2_4], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_4, answer_2_4] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing final sum Q+R, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_1[r].append(thinking)
            all_answer_3_1[r].append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Final answer synthesis and verification.", is_sub_task=True)
    agents.append(f"Final Decision agent, final sum Q+R calculation, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
