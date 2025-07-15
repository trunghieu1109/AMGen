async def forward_178(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: SC-CoT subtask_1.1
    cot_sc_instruction = (
        "Sub-task 1.1: Rigorously verify key properties of W, X, Y, and Z: "
        "1) check W†W for unitarity; "
        "2) compare X† to ±X; "
        "3) check Y’s Hermiticity, positive semidefiniteness, and trace=1; "
        "4) check Z’s Hermiticity. Provide detailed chain of thought and answer."
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1, possible_answers1 = [], []
    subtask1 = {"subtask_id":"subtask_1.1","instruction":cot_sc_instruction,"agent_collaboration":"SC_CoT"}
    for agent in cot_agents1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_thinking1, final_answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1.1: Synthesize the most consistent classification of W, X, Y, Z based on above thoughts and answers.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {final_thinking1.content}; answer - {final_answer1.content}")
    subtask1['response'] = {"thinking": final_thinking1, "answer": final_answer1}
    logs.append(subtask1)
    print("Step 1.1: ", sub_tasks[-1])

    # Stage 2: SC-CoT subtask_2.1
    cot_sc_instruction2 = (
        "Sub-task 2.1: Demonstrate that X is neither Hermitian nor skew-Hermitian and that e^X is not unitary. "
        "Provide a proof via eigenvalues or the series definition and compare (e^X)† to (e^X)⁻¹."
    )
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2, possible_answers2 = [], []
    subtask2 = {"subtask_id":"subtask_2.1","instruction":cot_sc_instruction2,"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking, answer = await agent([taskInfo, final_thinking1, final_answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_thinking2, final_answer2 = await final_decision_agent2(
        [taskInfo, final_thinking1, final_answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2.1: Synthesize the most consistent proof that X is not skew-Hermitian and e^X is non-unitary.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {final_thinking2.content}; answer - {final_answer2.content}")
    subtask2['response'] = {"thinking": final_thinking2, "answer": final_answer2}
    logs.append(subtask2)
    print("Step 2.1: ", sub_tasks[-1])

    # Stage 3: Debate subtask_3.1
    debate_instruction = (
        "Sub-task 3.1: Debate whether W and X represent the evolution operator of some quantum system. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask3 = {"subtask_id":"subtask_3.1","instruction":debate_instruction,"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent(
                    [taskInfo, final_thinking1, final_answer1, final_thinking2, final_answer2],
                    debate_instruction, r, is_sub_task=True
                )
            else:
                inputs = [taskInfo, final_thinking1, final_answer1, final_thinking2, final_answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking, answer = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking3[r].append(thinking)
            all_answer3[r].append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, final_thinking1, final_answer1, final_thinking2, final_answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3.1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask3)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs