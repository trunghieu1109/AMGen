async def forward_28(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, Subtask 1: Debate
    debate_instruction = (
        "Sub-task 1: Derive an analytic expression for the radius r of the circle of tangency "
        "between a torus (major radius 6, minor radius 3) and a sphere of radius 11, "
        "in terms of the distance D between their centers. "
        "Given solutions from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_agents = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    all_thinking1 = [[] for _ in range(self.max_round)]
    all_answer1 = [[] for _ in range(self.max_round)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1(
        [taskInfo] + all_thinking1[-1] + all_answer1[-1],
        "Sub-task 1: Derive an analytic expression for r. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0, Subtask 2: SC-CoT
    sc_instruction2 = (
        "Sub-task 2: Using R=6, a=3, and sphere radius 11, determine the two possible center-to-center distances "
        "D_i and D_o at which the torus rests on the sphere (external tangency conditions)."
    )
    N2 = self.max_sc
    sc_agents2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N2)
    ]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": sc_instruction2,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in sc_agents2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent solutions for D_i and D_o.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1, Subtask 3: SC-CoT
    sc_instruction3 = (
        "Sub-task 3: Substitute the computed distances D_i and D_o from Sub-task 2 into the analytic expression from Sub-task 1 "
        "to compute the tangency circle radii r_i and r_o."
    )
    N3 = self.max_sc
    sc_agents3 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N3)
    ]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": sc_instruction3,
        "context": [
            "user query", "thinking of subtask_1", "answer of subtask_1",
            "thinking of subtask_2", "answer of subtask_2"
        ],
        "agent_collaboration": "SC_CoT"
    }
    for agent in sc_agents3:
        thinking3, answer3 = await agent(
            [taskInfo, thinking1, answer1, thinking2, answer2],
            sc_instruction3,
            is_sub_task=True
        )
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings3.append(thinking3)
        possible_answers3.append(answer3)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3(
        [taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the most consistent values for r_i and r_o.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2, Subtask 4: Chain-of-Thought
    cot_instruction4 = (
        "Sub-task 4: Calculate the difference r_i minus r_o, express it as a reduced fraction m/n, and determine m+n."
    )
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3],
        cot_instruction4,
        is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction4,
        "context": [
            "user query", "thinking of subtask_1", "answer of subtask_1",
            "thinking of subtask_2", "answer of subtask_2",
            "thinking of subtask_3", "answer of subtask_3"
        ],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking4, "answer": answer4}
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs