async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Problem Restatement and Theoretical Conditions

    # Subtask 1: Formal restatement of problem conditions (Reflexion)
    reflexion_instruction_0_1 = (
        "Sub-task 1: Formally restate the problem conditions by precisely defining the divisibility condition p^2 | n^4 + 1 as the congruence n^4 ≡ -1 (mod p^2). "
        "Clearly specify that p is a prime and n is a positive integer. Avoid attempting to solve the congruence or make assumptions about n or p at this stage; focus solely on exact problem interpretation, notation, and clarifying implicit assumptions such as positivity and primality."
    )
    reflexion_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": reflexion_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_1, answer_0_1 = await reflexion_agent_0_1([taskInfo], reflexion_instruction_0_1, is_sub_task=True)
    agents.append(f"Reflexion agent {reflexion_agent_0_1.id}, restate problem conditions, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Subtask 2: Identify necessary conditions on prime p (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Identify and list necessary conditions on the prime p for the existence of n satisfying n^4 ≡ -1 (mod p^2). "
        "Derive conditions on p modulo small integers (e.g., p ≡ 1 (mod 8)) that allow -1 to be a quartic residue modulo p. "
        "Avoid assuming these conditions are sufficient; focus strictly on necessary theoretical constraints from number theory."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, identify necessary conditions on p, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Subtask 3: Explain rationale for searching primes and lifting solutions (SC_CoT)
    sc_cot_instruction_0_3 = (
        "Sub-task 3: Explain the rationale for searching primes in ascending order to find the minimal prime p satisfying the divisibility condition. "
        "Outline a theoretical approach to test the existence of n for each candidate prime p without exhaustive search, emphasizing the use of Hensel's lemma to lift solutions modulo p to modulo p^2. "
        "Avoid performing actual computations here; focus on the conceptual framework and strategy."
    )
    N_sc = self.max_sc
    sc_cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": sc_cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await sc_cot_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], sc_cot_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_cot_agents_0_3[i].id}, explain rationale for prime search and lifting, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent explanation for prime search and lifting." , is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize rationale for prime search and lifting, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Stage 1: Find roots modulo p and lift to modulo p^2

    # Subtask 1: For given prime p, find all roots n mod p with n^4 ≡ -1 mod p (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: For a given prime p, explicitly find and list all residue classes n modulo p satisfying n^4 ≡ -1 (mod p). "
        "Return the complete set of roots, not just their count. Verify each root programmatically by computing n^4 mod p to avoid manual arithmetic errors. "
        "Store this full root set in shared context for subsequent subtasks."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", answer_0_2.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, answer_0_2, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, find all roots modulo p, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Subtask 2: For each root modulo p, lift to modulo p^2 using Hensel's lemma and verify (SC_CoT)
    sc_cot_instruction_1_2 = (
        "Sub-task 2: For each root n modulo p found in Sub-task 1, apply Hensel's lemma to lift the solution uniquely to a root modulo p^2. "
        "Verify the correctness of each lifted root by computing (lifted_n)^4 mod p^2 and confirming it equals -1 mod p^2. "
        "Return the full set of lifted roots modulo p^2 with explicit verification results."
    )
    N_sc_1_2 = self.max_sc
    sc_cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": sc_cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await sc_cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], sc_cot_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_cot_agents_1_2[i].id}, lift roots modulo p to p^2, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent set of lifted roots modulo p^2." , is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize lifted roots modulo p^2, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Subtask 3: Identify minimal prime p for which lifting is possible (Reflexion)
    reflexion_instruction_1_3 = (
        "Sub-task 3: Identify the smallest prime p for which at least one root modulo p can be lifted to a root modulo p^2, confirming the existence of n such that p^2 divides n^4 + 1. "
        "Justify why smaller primes fail this lifting condition based on the results from subtasks 1 and 2. "
        "Provide a rigorous argument referencing the full sets of roots and lifts."
    )
    reflexion_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": reflexion_instruction_1_3,
        "context": ["user query", thinking_1_1.content, thinking_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await reflexion_agent_1_3([taskInfo, thinking_1_1, thinking_1_2], reflexion_instruction_1_3, is_sub_task=True)
    agents.append(f"Reflexion agent {reflexion_agent_1_3.id}, identify minimal prime p with lifting, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Stage 2: Find minimal positive integer m such that m^4 + 1 divisible by p^2

    # Subtask 1: For minimal prime p, examine all lifted roots modulo p^2, find least positive integer m for each, verify divisibility (CoT)
    cot_instruction_2_1 = (
        "Sub-task 1: Given the minimal prime p found in stage_1.subtask_3, examine all lifted roots modulo p^2 obtained in stage_1.subtask_2. "
        "For each lifted root, determine the least positive integer representative m modulo p^2 and verify that m^4 + 1 is divisible by p^2. "
        "Collect all such m values corresponding to the lifted roots."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, find least positive m for each lifted root, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Subtask 2: From candidate m values, identify minimal positive integer m globally and verify minimality (Reflexion)
    reflexion_instruction_2_2 = (
        "Sub-task 2: From the set of candidate integers m found in subtask_1, identify the minimal positive integer m such that m^4 + 1 is divisible by p^2. "
        "Verify globally that no smaller positive integer outside these lifted roots satisfies the congruence n^4 ≡ -1 (mod p^2). "
        "Provide a rigorous minimality proof or exhaustive verification within the modulus range to confirm correctness."
    )
    reflexion_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await reflexion_agent_2_2([taskInfo, thinking_2_1, answer_2_1], reflexion_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion agent {reflexion_agent_2_2.id}, identify minimal m globally and verify minimality, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
