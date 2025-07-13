async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Analyze the problem statement thoroughly and extract all relevant parameters and conditions. "
        "Summarize the key mathematical conditions: p is a prime number, n and m are positive integers, and the divisibility condition p^2 divides n^4 + 1 can be reformulated as the modular congruence n^4 ≡ -1 (mod p^2). "
        "Clarify that the problem reduces to finding primes p for which the congruence x^4 ≡ -1 (mod p^2) has a solution, and then finding the minimal such positive integer x (denoted m). "
        "Explicitly state assumptions such as no upper bounds on n or m, and that p > 1. Avoid assuming the existence of solutions for all primes without proof. "
        "This subtask sets the foundation for the entire problem and ensures a clear understanding of the problem's scope and constraints."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem statement, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: For each prime p in ascending order starting from 2, determine whether there exists an integer n such that n^4 ≡ -1 (mod p). "
        "Use properties of the multiplicative group modulo p and the order of elements to check solvability of x^4 ≡ -1 (mod p). "
        "Exclude primes for which no solution exists modulo p, as no solution modulo p^2 can exist. "
        "Document the reasoning and results clearly to identify candidate primes for the next step."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_2}")
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, checking solvability modulo p, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: For each candidate prime p identified in Sub-task 2, enumerate all solutions m_0 modulo p of x^4 ≡ -1 (mod p). "
        "For each such solution m_0, enumerate all candidate lifts x = m_0 + p * t for t = 0 to p-1. "
        "Compute x^4 + 1 modulo p^2 for each candidate to verify if p^2 divides x^4 + 1. "
        "Explicitly identify which primes admit at least one valid lift. "
        "This step requires concrete numeric modular exponentiation and verification, not just theoretical application of Hensel's lemma. "
        "Avoid premature acceptance of solutions without explicit verification and produce a list of primes with confirmed solutions modulo p^2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_3}")
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, enumerating and verifying lifts modulo p^2, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Identify the least prime p for which the congruence x^4 ≡ -1 (mod p^2) has a verified solution, based on the results of Sub-task 3. "
        "Document the reasoning and verification steps that confirm minimality of p. "
        "This subtask synthesizes the candidate primes and confirms the minimal prime p satisfying the problem's first condition."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_4}")
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, selecting minimal prime p, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: For the prime p found in Sub-task 4, find the least positive integer m such that m^4 ≡ -1 (mod p^2). "
        "Enumerate all candidate lifts from the solutions modulo p as in Sub-task 3, explicitly compute m^4 + 1 modulo p^2 for each candidate, "
        "and identify the minimal positive integer m satisfying the divisibility condition. "
        "Include explicit numeric verification and minimality checks, avoiding assumptions based solely on modulo p solutions. "
        "Output the minimal m with proof of correctness."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking1.content, answer1.content, thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_5}")
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, finding minimal m with explicit verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_6 = (
        "Sub-task 6: Verify the correctness of the identified prime p and integer m by explicitly checking that p^2 divides m^4 + 1. "
        "Additionally, confirm that no smaller prime or smaller positive integer satisfies the conditions. "
        "Include a thorough consistency check and validation of all previous results. "
        "Finally, synthesize the verified results into a clear, concise final answer to the problem, explicitly stating the values of p and m and the verification outcomes. "
        "This step ensures internal consistency and final correctness, integrating all prior computations and corrections. "
        + reflect_inst
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc_6 = {
        "subtask_id": "stage_1.subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before agent call: {subtask_desc_6}")
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, verifying and synthesizing final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6],
                                               "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
