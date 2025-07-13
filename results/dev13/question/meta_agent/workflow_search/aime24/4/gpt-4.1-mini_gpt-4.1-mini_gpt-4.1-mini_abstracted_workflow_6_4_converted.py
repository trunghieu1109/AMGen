async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Problem Domain and Theoretical Foundations

    # Subtask 1: Identify and clearly state the domain of the problem
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state the domain of the problem: all positive integers n and prime numbers p "
        "such that p^2 divides n^4 + 1. Emphasize the problem setting without attempting to solve any congruences or make assumptions about p."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, identifying problem domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Explain the modular arithmetic condition n^4 ≡ -1 (mod p^2) and its implications
    cot_instruction_0_2 = (
        "Sub-task 2: Explain the modular arithmetic condition n^4 ≡ -1 (mod p^2) and its implications for the existence of solutions. "
        "Focus on the meaning of this congruence and the challenges it poses, without attempting to find explicit solutions."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, explaining modular arithmetic condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Clarify assumptions about the prime p
    cot_instruction_0_3 = (
        "Sub-task 3: Clarify assumptions about the prime p, including why p=2 is unlikely to satisfy the condition and why p is assumed to be an odd prime. "
        "Avoid making unverified claims; provide reasoning based on modular arithmetic properties."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, clarifying prime assumptions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Subtask 4: Describe theoretical framework for lifting solutions modulo p to modulo p^2
    cot_sc_instruction_0_4 = (
        "Sub-task 4: Describe the theoretical framework for lifting solutions from modulo p to modulo p^2, including a precise statement of Hensel's lemma and its applicability conditions. "
        "Emphasize that Hensel's lemma guarantees existence and uniqueness of the lift but does not provide the explicit numeric value of the lifted root."
    )
    N_sc_0_4 = self.max_sc
    cot_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_4)]
    possible_answers_0_4 = []
    possible_thinkings_0_4 = []
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_sc_instruction_0_4,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_4):
        thinking_i, answer_i = await cot_agents_0_4[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_0_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_4[i].id}, describing lifting framework, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_4.append(answer_i)
        possible_thinkings_0_4.append(thinking_i)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + possible_answers_0_4 + possible_thinkings_0_4, "Sub-task 4: Synthesize and choose the most consistent explanation for lifting solutions modulo p to p^2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    # Stage 1: Existence of Solutions Modulo p and p^2, and Minimal Prime Identification

    # Subtask 1: For a given prime p, determine existence of n such that n^4 ≡ -1 (mod p)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: For a given prime p, determine whether there exists an integer n such that n^4 ≡ -1 (mod p). "
        "Provide explicit reasoning or computations to confirm the existence or non-existence of such n modulo p."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, checking existence of n mod p, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and confirm existence of solutions modulo p.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: For primes p where solutions modulo p exist, analyze lifting to modulo p^2
    cot_sc_instruction_1_2 = (
        "Sub-task 2: For primes p where solutions modulo p exist, analyze whether these solutions can be lifted to solutions modulo p^2. "
        "This includes setting up the lifting problem and theoretically confirming the possibility of lifting using Hensel's lemma conditions."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1, thinking_0_4, answer_0_4], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, analyzing lifting to p^2, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and confirm lifting possibility to modulo p^2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Enumerate primes ascendingly and identify least prime p with solution modulo p^2
    reflexion_instruction_1_3 = (
        "Sub-task 3: Enumerate primes in ascending order and apply the above tests to identify the least prime p for which there exists a positive integer n with n^4 + 1 divisible by p^2. "
        "Provide a clear, stepwise justification for the minimality of p."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": reflexion_instruction_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, reflexion_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, enumerating primes and identifying least p, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback, correct = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, reflexion_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining prime minimality, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Find minimal m for identified least prime p

    # Subtask 1: For identified least prime p, find integer a modulo p such that a^4 ≡ -1 (mod p)
    cot_sc_instruction_2_1 = (
        "Sub-task 1: For the identified least prime p, find the integer a modulo p such that a^4 ≡ -1 (mod p). "
        "This is the base root modulo p from which lifting will proceed. Provide explicit numeric value(s) of a."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, finding base root modulo p, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and provide explicit numeric base root modulo p.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Explicitly compute and verify the lifted solution modulo p^2
    cot_sc_instruction_2_2 = (
        "Sub-task 2: Explicitly compute and verify the lifted solution modulo p^2 using Hensel's lemma. "
        "Starting from the root a modulo p, calculate the derivative f'(a) modulo p, solve the linear congruence for k in f(a + p·k) ≡ 0 (mod p^2), "
        "and find the unique lifted root m = a + p·k modulo p^2. Alternatively, perform a brute-force search over candidates m = a + p·k for k = 0 to p-1 to find the minimal positive integer m satisfying m^4 + 1 ≡ 0 (mod p^2). "
        "Emphasize explicit numeric verification and avoid assuming the lifted root equals a."
    )
    N_sc_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_2):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, computing and verifying lifted root modulo p^2, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and provide explicit numeric lifted root modulo p^2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Subtask 3: Verify minimality of found integer m
    reflexion_instruction_2_3 = (
        "Sub-task 3: Verify that the found integer m is indeed the minimal positive integer such that m^4 + 1 is divisible by p^2. "
        "This includes checking all positive integers less than m for the divisibility condition and confirming no smaller integer satisfies it."
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_3 = self.max_round
    cot_inputs_2_3 = [taskInfo, thinking_2_2, answer_2_2]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": reflexion_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, reflexion_instruction_2_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, verifying minimality of m, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    for i in range(N_max_2_3):
        feedback, correct = await critic_agent_2_3([taskInfo, thinking_2_3, answer_2_3], "Please review and provide the limitations of provided minimality verification. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_3.extend([thinking_2_3, answer_2_3, feedback])
        thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, reflexion_instruction_2_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, refining minimality verification, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
