async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Enumerate all roots modulo candidate primes p
    cot_instruction_1 = (
        "Sub-task 1: Identify all primes p for which there exists a positive integer n such that p divides n^4 + 1, i.e., n^4 ≡ -1 (mod p). "
        "For each candidate prime p (including small primes 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97), enumerate all solutions x in {1, ..., p-1} satisfying x^4 ≡ -1 (mod p). "
        "Return the list of such primes p along with their complete root sets R = [r_1, r_2, ..., r_k]. "
        "Use number theory results to narrow candidates but verify roots computationally to avoid missing any."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, enumerated all roots modulo candidate primes, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Parse output from subtask 1: expect a structured list of primes and their roots
    # For example, answer1.content might be JSON or structured text listing primes and roots
    # We assume the agent returns a JSON string with format: [{"prime": p, "roots": [r1, r2, ...]}, ...]
    import json
    try:
        prime_roots_list = json.loads(answer1.content)
    except Exception:
        prime_roots_list = []

    # Stage 1 Sub-task 2: Verify for each candidate prime p and roots R whether p^2 divides n^4 + 1 for some n
    # including smaller primes like 5 and 13 explicitly
    cot_sc_instruction_2 = (
        "Sub-task 2: For each candidate prime p and its complete root set R from Sub-task 1, verify whether there exists a positive integer n such that p^2 divides n^4 + 1. "
        "Apply Hensel's lemma or direct modular arithmetic to lift each root r in R modulo p to a unique root modulo p^2, provided f'(r) mod p ≠ 0. "
        "For each lifted root, check if there exists an integer m ≡ ℓ (mod p^2) such that m^4 + 1 ≡ 0 (mod p^2). "
        "Search exhaustively over all such lifted roots and their cosets modulo p to find the minimal positive m for that p. "
        "Repeat this process for all candidate primes, including primes smaller than the initially identified p (such as 5, 13), to ensure no smaller prime satisfies the stronger condition. "
        "Return the least prime p for which such an m exists, along with the full set of lifted roots modulo p^2 and their minimal corresponding m values."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, verified lifting and minimal m search for candidate primes, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the least prime p and full lifted roots with minimal m satisfying p^2 | m^4 + 1.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Parse output from subtask 2: expect JSON with least prime p, lifted roots modulo p^2, and minimal m values
    try:
        verified_data = json.loads(answer2.content)
        least_prime_p = verified_data.get('least_prime_p')
        lifted_roots = verified_data.get('lifted_roots')  # list of roots mod p^2
        minimal_m_values = verified_data.get('minimal_m_values')  # list of minimal m per root
    except Exception:
        least_prime_p = None
        lifted_roots = []
        minimal_m_values = []

    # Stage 2 Sub-task 1: Given least prime p and lifted roots modulo p^2, find the least positive integer m
    cot_instruction_3 = (
        "Sub-task 1: Given the least prime p and the full set of lifted roots modulo p^2 from Stage 1 Sub-task 2, "
        "for each lifted root ℓ modulo p^2, enumerate all integers in the residue class ℓ + k·p (for k = 0 to p-1) to find the minimal positive integer m satisfying m^4 ≡ -1 (mod p^2). "
        "Compare all such minimal m values from all lifted roots and select the global minimum. "
        "Return the minimal m alongside the prime p."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2.content, answer2.content], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_3.id}, searched minimal m over all lifted roots, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Stage 2 Sub-task 2: Verify final results
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_4 = (
        "Sub-task 2: Verify the final results: confirm that the found prime p and integer m satisfy p^2 | m^4 + 1, and that p is indeed the smallest prime with this property. "
        "Also verify that m is the smallest positive integer with this property for that p. "
        "Perform alternative checks or proofs if possible (e.g., verifying no smaller prime or smaller m satisfies the conditions). "
        "Provide a final answer with detailed justification, including the full reasoning chain and verification outcomes. "
        + reflect_inst
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_4 = [taskInfo, thinking3.content, answer3.content, thinking2.content, answer2.content]
    subtask_desc4 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", thinking3.content, answer3.content, thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, initial verification, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4.content, answer4.content], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_4.extend([thinking4.content, answer4.content, feedback.content])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining verification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
