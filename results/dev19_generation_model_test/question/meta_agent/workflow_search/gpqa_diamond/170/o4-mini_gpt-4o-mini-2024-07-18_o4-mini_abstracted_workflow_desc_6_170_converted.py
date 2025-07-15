async def forward_170(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Classify substituents by activation/deactivation and directing effects
    cot_instruction = (
        "Sub-task 1: Identify and classify each substituent (–CH3, –C2H5, –Cl, –NO2, –COOC2H5, –COOH) "
        "by its electronic (activating/deactivating) and directing (ortho/para vs. meta) effects and estimate steric hindrance at ortho positions for the query."    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, classifying substituents, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Debate to rank by increasing para-isomer weight fraction
    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer.")
    debate_instruction_4 = (
        "Sub-task 4: Rank the six substrates by increasing weight fraction of the para-isomer and match the resulting order to one of the provided choices. "
        + debate_instr)
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking1, answer1], debate_instruction_4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking[r-1] + all_answer[r-1]
                thinking4, answer4 = await agent(inputs, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking[r].append(thinking4)
            all_answer[r].append(answer4)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    inputs_final = [taskInfo, thinking1, answer1] + all_thinking[-1] + all_answer[-1]
    thinking4, answer4 = await final_decision_agent(inputs_final, "Sub-task 4: " + final_instr, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs