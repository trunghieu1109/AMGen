async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Enumerate primes p in ascending order starting from the smallest primes. For each prime p, enumerate all positive integers n modulo p such that n^4 ≡ -1 (mod p). "
        "Document all such n values explicitly. Avoid assuming existence of solutions without explicit enumeration. This subtask sets the foundation for identifying candidate primes and candidate solutions modulo p."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, enumerating n modulo p for primes p, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: For each candidate solution n0 modulo p found in Sub-task 0.1, perform a brute-force search over k in [0, p-1] to check whether (n0 + k·p)^4 + 1 ≡ 0 (mod p^2). "
        "Explicitly verify modular arithmetic computations step-by-step and document all successes and failures. Record all valid lifted solutions modulo p^2."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, brute-force lifting verification modulo p^2, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 3: Implement Hensel's lemma to lift each solution n0 modulo p to a solution modulo p^2. Compute the correction term k using the formula k ≡ -f(n0)/(p·f'(n0)) mod p, where f(x) = x^4 + 1 and f'(x) = 4x^3. "
        "Calculate x = n0 + k·p and verify explicitly that x^4 + 1 ≡ 0 (mod p^2). Compare these lifted solutions with those found by brute-force in Sub-task 0.2 to cross-validate correctness. Document all modular arithmetic steps in detail."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, Hensel lifting and verification, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    debate_instruction_0_4 = (
        "Sub-task 4: Cross-validate the lifted solutions from Sub-task 0.3 with brute-force verified solutions from Sub-task 0.2. Identify any discrepancies and resolve them by re-computation or detailed modular arithmetic checks. Confirm the minimal positive integer m modulo p^2 among all valid lifted solutions. This subtask acts as an internal verifier to prevent propagation of arithmetic errors."
    )
    debate_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0_4 = self.max_round
    all_thinking_0_4 = [[] for _ in range(N_max_0_4)]
    all_answer_0_4 = [[] for _ in range(N_max_0_4)]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": debate_instruction_0_4,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_4):
        for i, agent in enumerate(debate_agents_0_4):
            if r == 0:
                thinking_0_4, answer_0_4 = await agent(
                    [taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3],
                    debate_instruction_0_4, r, is_sub_task=True
                )
            else:
                input_infos_0_4 = [taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3] + all_thinking_0_4[r-1] + all_answer_0_4[r-1]
                thinking_0_4, answer_0_4 = await agent(input_infos_0_4, debate_instruction_0_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating lifted solutions, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
            all_thinking_0_4[r].append(thinking_0_4)
            all_answer_0_4[r].append(answer_0_4)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4(
        [taskInfo] + all_thinking_0_4[-1] + all_answer_0_4[-1],
        "Sub-task 0.4: Synthesize and confirm minimal positive integer m modulo p^2",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Derive necessary and sufficient theoretical conditions for the existence of solutions to x^4 ≡ -1 (mod p^2). "
        "Analyze the multiplicative group modulo p and p^2, and characterize primes p for which -1 is a fourth power residue modulo p^2. Use number theory tools such as group theory and Hensel's lemma. "
        "Ensure that theoretical results align with enumerations and verifications from Stage 0. Avoid abstract reasoning without explicit connection to computational results."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_4, answer_0_4], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, deriving theoretical conditions, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1.1: Synthesize and choose the most consistent and correct theoretical conditions",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Formulate explicit, efficient algorithms or criteria based on the theoretical results from Sub-task 1.1 to test for the existence of solutions to x^4 ≡ -1 (mod p^2). "
        "Include step-by-step instructions for applying these criteria and verifying solutions. Emphasize that all candidate solutions must be explicitly verified modulo p^2 as per Stage 0 procedures."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, formulating criteria for solution existence, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 1.2: Synthesize and choose the most consistent and correct criteria for testing solutions",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Using the criteria and algorithms from Stage 1, apply a systematic search over primes in ascending order to identify the least prime p for which there exists a positive integer n such that p^2 divides n^4 + 1. "
        "For each prime, use the enumeration and lifting methods from Stage 0 to find and verify solutions modulo p^2. Document the minimal prime p found and all relevant modular arithmetic evidence supporting its minimality. Avoid skipping verification steps or assuming minimality without explicit checks."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i]([taskInfo, thinking_1_2, answer_1_2, thinking_0_4, answer_0_4], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, identifying minimal prime p, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + possible_answers_2_1 + possible_thinkings_2_1,
        "Sub-task 2.1: Synthesize and choose the minimal prime p with solution",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_instruction_2_2 = (
        "Sub-task 2: For the identified minimal prime p, find the least positive integer m such that m^4 + 1 is divisible by p^2. "
        "Use the verified lifted solutions from Stage 0 and confirm minimality by explicit modular arithmetic checks. Provide detailed computations showing m^4 + 1 ≡ 0 (mod p^2). If multiple solutions exist, ensure the smallest positive integer is selected. Include a brute-force verification fallback if needed to confirm minimality."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1, thinking_0_4, answer_0_4], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, finding minimal m for prime p, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instruction_2_3 = (
        "Sub-task 3: Perform an independent verification of the minimal prime p and minimal integer m found in Sub-tasks 2.1 and 2.2. "
        "Re-compute m^4 + 1 modulo p^2, check no smaller prime or integer satisfies the conditions, and confirm all modular arithmetic steps. "
        "This verification should be done by a separate agent or team to ensure unbiased validation. Document all verification steps and results explicitly."
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": debate_instruction_2_3,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking_2_3, answer_2_3 = await agent(
                    [taskInfo, thinking_2_1, answer_2_1, thinking_2_2, answer_2_2],
                    debate_instruction_2_3, r, is_sub_task=True
                )
            else:
                input_infos_2_3 = [taskInfo, thinking_2_1, answer_2_1, thinking_2_2, answer_2_2] + all_thinking_2_3[r-1] + all_answer_2_3[r-1]
                thinking_2_3, answer_2_3 = await agent(input_infos_2_3, debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying minimal prime and m, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
            all_thinking_2_3[r].append(thinking_2_3)
            all_answer_2_3[r].append(answer_2_3)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3(
        [taskInfo] + all_thinking_2_3[-1] + all_answer_2_3[-1],
        "Sub-task 2.3: Final verification of minimal prime p and minimal integer m",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Synthesize all results from previous stages into a final, clear, and concise answer. State the minimal prime p and minimal positive integer m such that m^4 + 1 is divisible by p^2. "
        "Provide a final explicit modular arithmetic demonstration, e.g., 'Check: m^4 + 1 ≡ 0 (mod p^2)' with numeric evidence. Confirm that no smaller prime or integer satisfies the problem conditions. Include a summary of the verification process and highlight how all previous errors have been addressed. This final step ensures correctness and completeness."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_3.content, answer_2_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent(
                    [taskInfo, thinking_2_3, answer_2_3],
                    debate_instruction_3_1, r, is_sub_task=True
                )
            else:
                input_infos_3_1 = [taskInfo, thinking_2_3, answer_2_3] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, synthesizing final answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1(
        [taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1],
        "Sub-task 3.1: Provide final verified answer for minimal prime p and minimal integer m",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
