async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Parse statements into structured claims (CoT)
    cot_instruction = (
        "Sub-task 1: Parse each of the four senior-scientist statements into structured chemical claims: "
        "identify the catalyst type, activator type, reaction step, and any economic considerations."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                             model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Evaluate each statement via Debate agents
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    N_round = self.max_round
    for sid, text in [
        ("2_1", "Such combined systems are already implemented on an industrial scale in the US."),
        ("2_2", "Aluminum-based activators do not work for the essential additional reaction step."),
        ("2_3", "One can use a catalyst of a group VIa transition metal in combination with specific activators."),
        ("2_4", "Certain noble metal catalysts can be used but are too expensive.")
    ]:
        debate_instruction = (
            f"Sub-task {sid}: Evaluate Statement {sid.split('_')[1]} ('{text}') with context from Sub-task 1. "
            "Given solutions to the problem from other agents, consider their opinions as additional advice. "
            "Please think carefully and provide an updated answer."
        )
        debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                       model=self.node_model, role=role, temperature=0.5)
                        for role in self.debate_role]
        all_thinking = [[] for _ in range(N_round)]
        all_answer = [[] for _ in range(N_round)]
        sub_desc = {
            "subtask_id": f"subtask_{sid}",
            "instruction": debate_instruction,
            "context": ["user query", thinking1, answer1],
            "agent_collaboration": "Debate"
        }
        for r in range(N_round):
            for i, agent in enumerate(debate_agents):
                if r == 0:
                    t, a = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
                else:
                    ctx = [taskInfo, thinking1, answer1] + all_thinking[r-1] + all_answer[r-1]
                    t, a = await agent(ctx, debate_instruction, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t.content}; answer: {a.content}")
                all_thinking[r].append(t)
                all_answer[r].append(a)
        final_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                   model=self.node_model, temperature=0.0)
        t_final, a_final = await final_agent(
            [taskInfo, thinking1, answer1] + all_thinking[-1] + all_answer[-1],
            f"Sub-task {sid}: Evaluate Statement {sid.split('_')[1]}" + final_instr,
            is_sub_task=True
        )
        agents.append(f"Final Decision agent {final_agent.id}, thinking: {t_final.content}; answer: {a_final.content}")
        sub_tasks.append(f"Sub-task {sid} output: thinking - {t_final.content}; answer - {a_final.content}")
        sub_desc['response'] = {"thinking": t_final, "answer": a_final}
        logs.append(sub_desc)
        print(f"Step {sid}: ", sub_tasks[-1])

    # Stage 3: Aggregate verdicts with SC-CoT
    cot_sc_instruction = (
        "Sub-task 3: Aggregate the four validation results into a classification table: "
        "map each statement to its verdict (‘supported’, ‘refuted’, or ‘unknown’) with concise evidence summaries."
    )
    N_sc = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.5)
                 for _ in range(N_sc)]
    possible_think = []
    possible_ans = []
    sub_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction,
        "context": ["user query"] + [logs[i]['response']['thinking'] for i in range(1, 5)] + [logs[i]['response']['answer'] for i in range(1, 5)],
        "agent_collaboration": "SC_CoT"
    }
    for agent in sc_agents:
        t_sc, a_sc = await agent([taskInfo] + possible_think + possible_ans, cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t_sc.content}; answer: {a_sc.content}")
        possible_think.append(t_sc)
        possible_ans.append(a_sc)
    final_sc = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                            model=self.node_model, temperature=0.0)
    t3f, a3f = await final_sc([taskInfo] + possible_think + possible_ans,
                              "Sub-task 3: Synthesize and choose the most consistent and correct classification table.",
                              is_sub_task=True)
    agents.append(f"Final Decision agent {final_sc.id}, thinking: {t3f.content}; answer: {a3f.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {t3f.content}; answer - {a3f.content}")
    sub_desc3['response'] = {"thinking": t3f, "answer": a3f}
    logs.append(sub_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Debate final selection
    debate_instr4 = (
        "Sub-task 4: Based on the aggregated classifications, select the single statement conclusively supported and provide final justification. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    all_t4 = [[] for _ in range(N_round)]
    all_a4 = [[] for _ in range(N_round)]
    sub_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr4,
        "context": ["user query", t3f, a3f],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round):
        for agent in debate_agents4:
            if r == 0:
                t4, a4 = await agent([taskInfo, t3f, a3f], debate_instr4, r, is_sub_task=True)
            else:
                ctx = [taskInfo, t3f, a3f] + all_t4[r-1] + all_a4[r-1]
                t4, a4 = await agent(ctx, debate_instr4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t4.content}; answer: {a4.content}")
            all_t4[r].append(t4)
            all_a4[r].append(a4)
    final4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                           model=self.node_model, temperature=0.0)
    t4f, a4f = await final4(
        [taskInfo, t3f, a3f] + all_t4[-1] + all_a4[-1],
        "Sub-task 4: Final selection based on aggregated classifications." + final_instr,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final4.id}, thinking: {t4f.content}; answer: {a4f.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {t4f.content}; answer - {a4f.content}")
    sub_desc4['response'] = {"thinking": t4f, "answer": a4f}
    logs.append(sub_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(t4f, a4f, sub_tasks, agents)
    return final_answer, logs