async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Modular Condition and Prime Condition (Debate)
    debate_instr_1 = (
        "Sub-task 1: Identify and clearly state the modular condition for n^4 + 1 to be divisible by a prime p, "
        "i.e., establish that n^4 ≡ -1 (mod p). Carefully explain why this condition is necessary and what it implies about the existence of n modulo p. "
        "Avoid assuming any particular prime or solution at this stage."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0)
        for role in self.debate_role
    ]
    N_max_1 = self.max_round

    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]

    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1[r].append(thinking)
            all_answer_1[r].append(answer)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1],
                                                      "Sub-task 1: Synthesize and choose the most consistent answer for modular condition and prime condition." +
                                                      " Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Search for n modulo p, compute a, apply Hensel lifting, and verify (SC_CoT + Reflexion merged)
    cot_sc_instruction_2 = (
        "Sub-task 2: For primes p satisfying the condition from Sub-task 1, explicitly search for positive integers n modulo p such that n^4 ≡ -1 (mod p). "
        "For each candidate n, compute a = (n^4 + 1)/p modulo p. Then apply Hensel's lemma to lift n modulo p to m modulo p^2 satisfying m^4 + 1 ≡ 0 (mod p^2). "
        "Explicitly compute the correction term k in m = n + p*k by solving the linear congruence derived from f'(n) = 4n^3 modulo p. "
        "Verify that the lifted solution m satisfies the congruence modulo p^2 exactly. Document all numeric values (p, n, a, k, m) and verification results. "
        "Avoid assumptions and ensure numeric correctness."
    )
    N_sc = self.max_sc
    cot_sc_agents = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        for _ in range(N_sc)
    ]

    possible_answers_2 = []
    possible_thinkings_2 = []

    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking, answer = await cot_sc_agents[i]([taskInfo, thinking_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, searching n and lifting solution, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2.append(answer)
        possible_thinkings_2.append(thinking)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2,
                                                      "Sub-task 2: Synthesize and choose the most consistent and correct lifted solutions with numeric verification." +
                                                      " Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Verify minimality of p and find minimal m (Debate + SC_CoT)
    debate_instr_3 = (
        "Sub-task 3: Systematically test all primes smaller than candidate p found in Sub-task 2 for existence of n and m satisfying divisibility by p^2. "
        "Ensure minimality of p rigorously. Then, for confirmed minimal prime p, find the least positive integer m such that p^2 divides m^4 + 1. "
        "Avoid assumptions and verify all numeric computations. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0)
        for role in self.debate_role
    ]
    N_max_3 = self.max_round

    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]

    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", thinking_2.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2] + all_thinking_3[r-1]
                thinking, answer = await agent(input_infos, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3[r].append(thinking)
            all_answer_3[r].append(answer)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo, thinking_2] + all_thinking_3[-1],
                                                      "Sub-task 3: Synthesize and confirm minimal prime p and find minimal m." +
                                                      " Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
