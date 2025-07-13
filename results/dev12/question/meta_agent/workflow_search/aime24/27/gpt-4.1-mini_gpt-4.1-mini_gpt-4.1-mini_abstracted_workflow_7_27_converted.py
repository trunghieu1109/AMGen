async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: Define digits and formulate modular conditions
    # Sub-task 1: Define digits and problem constraints (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Define the digits of N as d1, d2, d3, d4 with d1 != 0. "
        "State clearly the problem constraints: changing any one digit of N to 1 results in a number divisible by 7. "
        "This applies independently to each digit replacement, regardless of the original digit value. "
        "Avoid assumptions about digits other than the leading digit constraint."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining digits and constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Sub-task 2: Formulate explicit modular arithmetic conditions (CoT)
    cot_instruction_2 = (
        "Sub-task 2: Formulate explicit modular arithmetic conditions for each digit replacement: "
        "For each position i (1 to 4), express the divisibility by 7 of the number formed by replacing d_i with 1 as a congruence modulo 7. "
        "Represent these congruences symbolically with clear coefficients and variables, avoiding ambiguous natural language descriptions."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating modular conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Sub-task 3: Solve system of modular congruences symbolically (CoT)
    cot_instruction_3 = (
        "Sub-task 3: Solve the system of modular congruences symbolically and deterministically to derive explicit relationships or restrictions on digits d1, d2, d3, d4. "
        "Use zero-temperature Chain-of-Thought reasoning to ensure consistent and correct derivations. Avoid partial or inconsistent solutions. "
        "Output the modular equations and the derived digit constraints explicitly as symbolic expressions or modular equalities."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, solving modular system, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # --------------------------------------------------------------------------------------------------------------
    # Stage 2: Enumerate candidates and verify divisibility
    # Sub-task 4: Enumerate all four-digit numbers satisfying modular constraints (SC_CoT)
    cot_sc_instruction_4 = (
        "Sub-task 4: Enumerate all four-digit numbers N = d1 d2 d3 d4 with d1 != 0 and digits 0-9 that satisfy the modular constraints derived in Sub-task 3. "
        "Use a controlled exhaustive search over the digit space constrained by the modular relationships. Generate a candidate list for further verification. "
        "Output the candidate numbers explicitly as a list."
    )
    N_sc = max(2, self.max_sc)  # Use at least 2 SC agents for consistency
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, enumerating candidates, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, 
                                                    "Sub-task 4: Synthesize and choose the consistent candidate list from enumerations.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing candidate list, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    # Parse candidate list from answer4 (assumed to be a list of integers in string form)
    import re
    candidate_numbers = list(map(int, re.findall(r'\b\d{4}\b', answer4.content)))

    # Sub-task 5: Verify each candidate rigorously (CoT)
    cot_instruction_5 = (
        "Sub-task 5: For each candidate number from Sub-task 4, rigorously verify the core divisibility condition: "
        "replace each digit of N one at a time with 1, and check if the resulting number is divisible by 7. "
        "Only retain candidates passing all four divisibility tests. Document verification results explicitly. "
        "Output the verified candidate list explicitly."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }

    # Prepare input with candidate list for verification
    candidate_list_str = ', '.join(map(str, candidate_numbers))
    input_for_5 = f"Candidates: {candidate_list_str}"
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4, input_for_5], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, verifying candidates, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    # Parse verified candidates from answer5
    verified_candidates = list(map(int, re.findall(r'\b\d{4}\b', answer5.content)))

    # --------------------------------------------------------------------------------------------------------------
    # Stage 3: Select greatest candidate and compute final answer
    # Sub-task 6: Identify greatest verified candidate (CoT)
    cot_instruction_6 = (
        "Sub-task 6: From the verified candidates in Sub-task 5, identify the greatest four-digit number N satisfying all conditions. "
        "Clearly justify the selection based on verification results and magnitude comparison."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    verified_candidates_str = ', '.join(map(str, verified_candidates))
    input_for_6 = f"Verified candidates: {verified_candidates_str}"
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content, answer5.content, input_for_6],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, input_for_6], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting greatest candidate, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    # Extract greatest candidate N from answer6
    import re
    greatest_candidate_match = re.search(r'\b\d{4}\b', answer6.content)
    if greatest_candidate_match:
        N = int(greatest_candidate_match.group())
    else:
        N = None

    # Sub-task 7: Compute Q and R from N (CoT)
    cot_instruction_7 = (
        "Sub-task 7: Compute Q and R where Q is the quotient and R the remainder when N is divided by 1000, i.e., Q = d1 and R = 100*d2 + 10*d3 + d4. "
        "Ensure that Q and R correspond to the verified greatest candidate."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    input_for_7 = f"Greatest candidate N: {N}"
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content, answer6.content, input_for_7],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6, input_for_7], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, computing Q and R, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)

    # Sub-task 8: Calculate and output Q + R (Reflexion)
    reflect_inst_8 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_8 = (
        "Sub-task 8: Calculate and output the sum Q + R as the final answer. "
        "Confirm that this result is consistent with all previous verification steps and modular constraints. "
        + reflect_inst_8
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    input_for_8 = f"Q and R info: {answer7.content}"
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", thinking7.content, answer7.content, input_for_8],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7, input_for_8], cot_reflect_instruction_8, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, calculating final sum Q+R, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
