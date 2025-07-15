async def forward_163(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Extract and classify parameters (SC_CoT)
    sc_instruction1 = (
        "Sub-task 1: Extract and classify observational parameters: orbital periods P1 and P2, "
        "radial velocity amplitudes K1a, K1b, K2a, K2b, and the definition of system mass from the user query."
    )
    N1 = self.max_sc
    sc_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": sc_instruction1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1_i, answer1_i = await sc_agents1[i](
            [taskInfo], sc_instruction1, is_sub_task=True
        )
        agents.append(
            f"CoT-SC agent {sc_agents1[i].id}, extracting parameters, thinking: {thinking1_i.content}; answer: {answer1_i.content}"
        )
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                         model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent parameter extraction.",
        is_sub_task=True
    )
    agents.append(
        f"Final Decision Agent {final_decision_agent1.id}, synthesizing parameter extraction, thinking: {thinking1.content}; answer: {answer1.content}"
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Derive formula (SC_CoT)
    sc_instruction2 = (
        "Sub-task 2: Derive formula for the ratio of total masses M_total,1/M_total,2 in terms of "
        "P1, P2, (K1a+K1b) and (K2a+K2b) using Kepler's third law and radial-velocity relations."
    )
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": sc_instruction2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents2[i](
            [taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True
        )
        agents.append(
            f"CoT-SC agent {sc_agents2[i].id}, deriving formula, thinking: {thinking2_i.content}; answer: {answer2_i.content}"
        )
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent formula.",
        is_sub_task=True
    )
    agents.append(
        f"Final Decision Agent {final_decision_agent2.id}, synthesizing formula, thinking: {thinking2.content}; answer: {answer2.content}"
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Compute numerical value (CoT)
    cot_instruction3 = (
        "Sub-task 3: Compute the numerical value of the mass ratio M_total,1/M_total,2 using the formula derived in Sub-task 2 "
        "and the values P1=2, P2=1, K1a+K1b=15, K2a+K2b=25."
    )
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                               model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent3(
        [taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True
    )
    agents.append(
        f"CoT agent {cot_agent3.id}, computing numerical ratio, thinking: {thinking3.content}; answer: {answer3.content}"
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compare to MCQ (Debate)
    debate_instruction4 = (
        "Sub-task 4: Compare the computed mass ratio to the provided multiple-choice options and select the closest match. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent",
                                   model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_i, answer4_i = await agent(
                    [taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True
                )
            else:
                inputs4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(
                    inputs4, debate_instruction4, r, is_sub_task=True
                )
            agents.append(
                f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}"
            )
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(
        f"Final Decision Agent {final_decision_agent4.id}, selecting final choice, thinking: {thinking4.content}; answer: {answer4.content}"
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
