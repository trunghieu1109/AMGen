async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Analyze divisibility conditions and find least prime p

    # Sub-task 1: Analyze divisibility condition p^2 | n^4 + 1
    cot_instruction_1 = (
        "Sub-task 1: Analyze the divisibility condition p^2 divides n^4 + 1 to derive necessary modular constraints on prime p and integer n. "
        "Specifically, determine the conditions under which the congruence n^4 ≡ -1 (mod p) has solutions and when these solutions can be lifted modulo p^2. "
        "Include characterization of primes p for which n^4 ≡ -1 (mod p) is solvable, understand multiplicity conditions for p^2 dividing n^4 + 1, and verify lifting conditions using Hensel's lemma. "
        "Avoid assumptions such as triviality of p=2 without verification. Emphasize full modular verification."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing divisibility condition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Sub-task 2: Identify least prime p satisfying conditions from subtask 1
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis from Sub-task 1, systematically check primes in ascending order to find the least prime p for which there exists a positive integer n such that p^2 divides n^4 + 1. "
        "For each candidate prime, verify existence of solutions modulo p and verify lifting modulo p^2 using criteria from Sub-task 1. Document verification for each prime explicitly. Avoid skipping primes or relying on heuristics without proof."
    )
    N_sc = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, checking primes for p, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the least prime p satisfying conditions." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Extract prime p from answer2 for next stages
    # (Assuming answer2.content contains the prime p explicitly, parse it)
    # For safety, parse prime p from answer2.content
    import re
    prime_p_match = re.search(r"\b(\d+)\b", answer2.content)
    if prime_p_match:
        prime_p = int(prime_p_match.group(1))
    else:
        prime_p = None

    # Stage 1: Find all roots modulo p and lift solutions modulo p^2

    # Sub-task 3: Find all solutions n0 modulo p to n^4 + 1 ≡ 0 (mod p), then apply Hensel's lemma to lift solutions modulo p^2
    debate_instruction_3 = (
        f"Sub-task 3: For the identified prime p={prime_p}, find all solutions n0 modulo p to the congruence n^4 + 1 ≡ 0 (mod p). "
        "Enumerate these roots explicitly. Then apply Hensel's lemma to each root n0 to lift solutions modulo p^2 by solving the linear congruence for the lifting parameter t in m = n0 + p*t. "
        "Document the derivation of the linear congruence, solve it explicitly, and produce candidate lifted solutions m modulo p^2. "
        "Avoid assuming trivial lifts or skipping the linear congruence step. This subtask produces a full set of candidates for m modulo p^2. "
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, lifting roots and generating candidates, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Synthesize candidate lifted solutions for m modulo p^2." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Stage 1.5: Verification and minimality check of candidate lifts

    # Sub-task 3.5: Enumerate all candidate lifts m = n0 + p*t for t in [0, p-1], compute m^4 + 1 mod p^2, tabulate results, identify minimal m
    reflexion_instruction_3_5 = (
        f"Sub-task 3.5: Given the candidate lifted solutions m modulo p^2 from Sub-task 3, enumerate all possible lifts m = n0 + p*t for t = 0 to p-1 for each root n0. "
        "Compute m^4 + 1 modulo p^2 explicitly for each candidate, tabulate the results, and identify which candidates satisfy p^2 divides m^4 + 1. "
        "Select the minimal positive integer m among them. Provide a clear verification table to expose any arithmetic slips and ensure correctness. "
        "If multiple candidates satisfy the condition, confirm minimality by direct comparison. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_5 = self.max_round
    cot_inputs_3_5 = [taskInfo, thinking3, answer3]
    subtask_desc3_5 = {
        "subtask_id": "subtask_3.5",
        "instruction": reflexion_instruction_3_5,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Debate | Reflexion"
    }
    thinking3_5, answer3_5 = await cot_agent_3_5(cot_inputs_3_5, reflexion_instruction_3_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_5.id}, verifying candidate lifts, thinking: {thinking3_5.content}; answer: {answer3_5.content}")
    for i in range(N_max_3_5):
        feedback, correct = await critic_agent_3_5([taskInfo, thinking3_5, answer3_5], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_5.extend([thinking3_5, answer3_5, feedback])
        thinking3_5, answer3_5 = await cot_agent_3_5(cot_inputs_3_5, reflexion_instruction_3_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_5.id}, refining verification, thinking: {thinking3_5.content}; answer: {answer3_5.content}")
    sub_tasks.append(f"Sub-task 3.5 output: thinking - {thinking3_5.content}; answer - {answer3_5.content}")
    subtask_desc3_5['response'] = {"thinking": thinking3_5, "answer": answer3_5}
    logs.append(subtask_desc3_5)

    # Stage 2: Final confirmation of minimal m

    # Sub-task 4: Using verified minimal m and prime p, perform final confirmation by direct substitution and modular arithmetic check
    reflexion_instruction_4 = (
        f"Sub-task 4: Using the verified minimal positive integer m from Sub-task 3.5 and the prime p={prime_p}, perform a final confirmation by direct substitution and modular arithmetic check that p^2 divides m^4 + 1. "
        "Document the substitution, compute m^4 + 1 modulo p^2 explicitly, and confirm divisibility. Provide the final answer m alongside detailed verification results to ensure completeness and accuracy. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3_5, answer3_5]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": reflexion_instruction_4,
        "context": ["user query", thinking3_5.content, answer3_5.content],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflexion_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, final confirmation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflexion_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final confirmation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
