async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Foundational Analysis and Theory

    # Subtask 1: Formulate modular condition n^4 + 1 ≡ 0 (mod p^2) as n^4 ≡ -1 (mod p^2)
    cot_instruction_0_1 = (
        "Sub-task 1: Formulate the modular congruence condition n^4 + 1 ≡ 0 (mod p^2) as n^4 ≡ -1 (mod p^2). "
        "Explain implications for existence of solutions modulo p and p^2, emphasizing the need to find roots of x^4 ≡ -1 modulo p and p^2. "
        "Avoid assuming existence of solutions without verification."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formulating modular condition, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Analyze necessary conditions on prime p for x^4 ≡ -1 (mod p) to have solutions
    cot_instruction_0_2 = (
        "Sub-task 2: Analyze necessary conditions on prime p for the congruence x^4 ≡ -1 (mod p) to have solutions. "
        "Explain the role of quartic residues modulo p, the multiplicative group structure modulo p, and the significance of primes p ≡ 1 (mod 8). "
        "Avoid assuming any prime without checking these conditions."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing prime conditions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Systematic search for roots x modulo p with x^4 ≡ -1 (mod p)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: For a given prime p, systematically search all x in [1, p-1] to find roots satisfying x^4 ≡ -1 (mod p). "
        "Verify each candidate root numerically. If no roots exist, conclude no solution modulo p and skip lifting for that prime. "
        "Record all valid roots."
    )
    N_sc = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, searching roots modulo p, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent and correct roots modulo p.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Explain Hensel lifting method from modulo p to modulo p^2
    cot_instruction_0_4 = (
        "Sub-task 4: Explain the method of lifting solutions from modulo p to modulo p^2 using Hensel's lemma. "
        "Detail the formula t ≡ -f(a)/p · (f'(a))^{-1} mod p for lifting a root a modulo p to modulo p^2. "
        "Emphasize computing derivative f'(x) = 4x^3 mod p and verifying invertibility. Avoid assuming uniqueness without verification."
    )
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_0_4,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_4, answer_0_4 = await cot_agent_0_4([taskInfo, thinking_0_3], cot_instruction_0_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_4.id}, explaining Hensel lifting, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 5: Strategy to search least prime p ≡ 1 (mod 8) with root lifting
    reflexion_instruction_0_5 = (
        "Sub-task 5: Establish a strategy to search for the least prime p ≡ 1 (mod 8) such that there exists at least one root x modulo p with x^4 ≡ -1 (mod p), "
        "and such root can be lifted to modulo p^2 using Hensel's lemma. Include explicit verification steps for root existence modulo p and successful lifting modulo p^2. "
        "Avoid hard-coding or assuming roots; generalize the search and verification."
    )
    cot_agent_0_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_0_5 = self.max_round
    cot_inputs_0_5 = [taskInfo, thinking_0_4]
    subtask_desc_0_5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_0_5,
        "context": ["user query", thinking_0_4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_5, answer_0_5 = await cot_agent_0_5(cot_inputs_0_5, reflexion_instruction_0_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_5.id}, establishing prime search strategy, thinking: {thinking_0_5.content}; answer: {answer_0_5.content}")
    for i in range(N_max_0_5):
        feedback_0_5, correct_0_5 = await critic_agent_0_5([taskInfo, thinking_0_5], "Please review and provide limitations. If correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_5.id}, feedback: {feedback_0_5.content}; correct: {correct_0_5.content}")
        if correct_0_5.content == "True":
            break
        cot_inputs_0_5.extend([thinking_0_5, feedback_0_5])
        thinking_0_5, answer_0_5 = await cot_agent_0_5(cot_inputs_0_5, reflexion_instruction_0_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_5.id}, refining prime search strategy, thinking: {thinking_0_5.content}; answer: {answer_0_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_0_5.content}; answer - {answer_0_5.content}")
    subtask_desc_0_5['response'] = {"thinking": thinking_0_5, "answer": answer_0_5}
    logs.append(subtask_desc_0_5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 1: Search for least prime p and roots modulo p and p^2

    # Subtask 1: Implement search over primes p ≡ 1 (mod 8), find roots modulo p and lift to p^2
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Search primes p ≡ 1 (mod 8) in ascending order. For each p, find all roots x modulo p with x^4 ≡ -1 (mod p). "
        "Apply Hensel lifting to each root to find roots modulo p^2. Verify correctness of all computations. Record all valid lifted roots for each prime."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_5], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, searching primes and lifting roots, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Identify least prime p and valid lifted roots.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 2: Verify existence of positive integer n with n^4 + 1 ≡ 0 (mod p^2) for found p
    cot_instruction_1_2 = (
        "Sub-task 2: Verify existence of at least one positive integer n satisfying n^4 + 1 ≡ 0 (mod p^2) for the identified least prime p. "
        "Confirm correctness of modular arithmetic and validity of lifted roots. Identify all minimal positive representatives modulo p^2."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, verifying existence of n, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 7: ", sub_tasks[-1])

    # Stage 2: Find minimal positive integer m such that m^4 + 1 divisible by p^2

    # Subtask 1: Determine least positive integer m from lifted roots modulo p^2
    cot_sc_instruction_2_1 = (
        "Sub-task 1: For the prime p found, determine the least positive integer m such that m^4 + 1 is divisible by p^2. "
        "Use lifted roots modulo p^2 and find minimal positive representatives. Avoid brute force over large ranges; rely on modular arithmetic and explicit verification."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, determining minimal m, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Find minimal positive m.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 8: ", sub_tasks[-1])

    # Subtask 2: Confirm minimality of m by checking all positive integers less than m
    reflexion_instruction_2_2 = (
        "Sub-task 2: Confirm minimality of m by checking all positive integers less than m to ensure none satisfy m^4 + 1 ≡ 0 (mod p^2). "
        "Include explicit modular arithmetic verification for each candidate. Update m if smaller integer found."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflexion_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, confirming minimality of m, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review and provide limitations. If correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflexion_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining minimality confirmation, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
