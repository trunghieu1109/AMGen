async def forward_17(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 0: Derive and validate a single, concise algebraic expression for the nonlinear constraint "
        "S = a^2b + a^2c + b^2a + b^2c + c^2a + c^2b in terms of symmetric polynomials of (a,b,c). "
        "Rewrite S using symmetric sums (a+b+c, ab+bc+ca, abc) to obtain a simplified formula such as S = 300q - 3r, "
        "where q = ab+bc+ca and r = abc. Confirm equivalence and maintain symmetry and nonnegativity constraints."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving algebraic representation, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)

    N_sc_0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc0_sc = {
        "subtask_id": "subtask_0_sc",
        "instruction": cot_instruction_0,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking_sc, answer_sc = await cot_agents_0[i]([taskInfo, thinking0, answer0], cot_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, validating algebraic form, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_0.append(answer_sc)
        possible_thinkings_0.append(thinking_sc)

    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_final, answer0_final = await final_decision_agent_0(
        [taskInfo] + possible_answers_0 + possible_thinkings_0,
        "Sub-task 0: Synthesize and choose the most consistent algebraic representation for the nonlinear constraint.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0 final output: thinking - {thinking0_final.content}; answer - {answer0_final.content}")
    subtask_desc0_sc['response'] = {"thinking": thinking0_final, "answer": answer0_final}
    logs.append(subtask_desc0_sc)

    print("Step 0: ", sub_tasks[-1])

    reflexion_instruction_1 = (
        "Sub-task 1: Using the formula from Stage 0, perform a detailed case analysis on the key parameter equation "
        "100q - r = 2,000,000 derived from S = 6,000,000 and a+b+c=300. Explicitly split into two cases: "
        "(a) r = 0, implying q = 20,000, and (b) r > 0, with bounds 0 < r ≤ 1,000,000 and 20,000 ≤ q ≤ 30,000. "
        "For case (a), analyze the quadratic system ab = 20,000 and a + b = 300 to find all valid (a,b,c) triples with one zero component. "
        "For case (b), identify feasible integer values of (q,r) that satisfy the equation and explore their implications for (a,b,c). "
        "Avoid overlooking any cases, especially r=0, and ensure all integer and nonnegativity constraints are respected. "
        "Produce explicit parameter sets and conditions to guide enumeration."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_reflect_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking0_final, answer0_final]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": reflexion_instruction_1,
        "context": ["user query", "thinking of subtask 0 final", "answer of subtask 0 final"],
        "agent_collaboration": "Reflexion"
    }
    thinking1, answer1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, detailed case analysis, thinking: {thinking1.content}; answer: {answer1.content}")

    for i in range(N_reflect_1):
        feedback, correct = await critic_agent_1([taskInfo, thinking1, answer1],
                                               "Please review and provide limitations or errors in the derivation. If correct, output exactly 'True'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1.extend([thinking1, answer1, feedback])
        thinking1, answer1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining case analysis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Enumerate all ordered triples (a,b,c) of nonnegative integers with a ≤ b ≤ c satisfying a + b + c = 300 "
        "and the nonlinear constraint S = 6,000,000, using the parameter conditions derived in Stage 1. "
        "Implement pruning strategies based on algebraic bounds and parameter restrictions to reduce the search space. "
        "For each ordered triple found, count the number of distinct permutations to obtain the total number of ordered triples. "
        "Ensure no duplicates or invalid triples are included. Use at least two independent enumeration methods (e.g., direct enumeration with pruning and algebraic reasoning) to cross-validate results. "
        "Provide a detailed enumeration report including counts per case (r=0 and r>0)."
    )
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 0 final", "answer of subtask 0 final"],
        "agent_collaboration": "SC_CoT | Debate"
    }
    for i in range(N_sc_2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1, thinking0_final, answer0_final], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, enumerating triples, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_final, answer2_final = await final_decision_agent_2(
        [taskInfo] + possible_answers_2 + possible_thinkings_2,
        "Sub-task 2: Synthesize and provide the most consistent enumeration results.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2['response'] = {"thinking": thinking2_final, "answer": answer2_final}
    logs.append(subtask_desc2)

    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 3: Perform rigorous verification and reconciliation of enumeration results from Stage 2. "
        "Critically analyze discrepancies between enumeration methods, verify each candidate triple satisfies both constraints exactly, "
        "and confirm correctness of permutation counting. Synthesize the final verified count of all ordered triples (a,b,c) meeting the problem conditions. "
        "Provide a clear final answer with a detailed verification report explaining reasoning, checks, and conflict resolution."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 0 final", "answer of subtask 0 final"],
        "agent_collaboration": "Debate | SC_CoT"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2_final, answer2_final, thinking1, answer1, thinking0_final, answer0_final], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2_final, answer2_final, thinking1, answer1, thinking0_final, answer0_final] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verification and reconciliation, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_final, answer3_final = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1],
                                                                 "Sub-task 3: Provide the final verified count and detailed verification report.",
                                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, final verification and synthesis, thinking: {thinking3_final.content}; answer: {answer3_final.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    subtask_desc3['response'] = {"thinking": thinking3_final, "answer": answer3_final}
    logs.append(subtask_desc3)

    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_final, answer3_final, sub_tasks, agents)
    return final_answer, logs
