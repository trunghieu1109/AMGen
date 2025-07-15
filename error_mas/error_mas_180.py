import asyncio
from collections import Counter

class SolarNeutrinoWorkflow:
    def __init__(self, node_model, debate_role, max_sc, max_round):
        self.node_model = node_model
        self.debate_role = debate_role
        self.max_sc = max_sc
        self.max_round = max_round

    async def forward_180(self, taskInfo):
        sub_tasks = []
        agents = []
        logs = []

        # Sub-task 1: Collect differential flux spectra via Chain-of-Thought
        cot_instruction = (
            "Sub-task 1: Collect numerical or analytic approximations of dΦ/dE for each pp-chain branch "
            "(pp-I, pep, 7Be, B-8) across 600–1000 keV. Provide data structures mapping energy to flux density."
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

        # Sub-task 2: Numerically integrate each branch spectrum via Debate
        debate_instr = (
            "Sub-task 2: Integrate each branch's spectrum from 700–800 keV and 800–900 keV to compute "
            "Φ_branch_700_800 and Φ_branch_800_900. Provide numeric results for all branches." 
            "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        )
        debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent",
                                        model=self.node_model, role=role, temperature=0.5)
                         for role in self.debate_role]
        all_thinking2 = []
        all_answer2 = []
        subtask_desc2 = {
            "subtask_id": "subtask_2",
            "instruction": debate_instr,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
            "agent_collaboration": "Debate"
        }
        for r in range(self.max_round):
            round_think = []
            round_ans = []
            for agent in debate_agents:
                if r == 0:
                    thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instr, r, is_sub_task=True)
                else:
                    inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                    thinking2, answer2 = await agent(inputs, debate_instr, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
                round_think.append(thinking2)
                round_ans.append(answer2)
            all_thinking2.append(round_think)
            all_answer2.append(round_ans)
        final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                             model=self.node_model, temperature=0.0)
        final_instr2 = (
            "Given all the above thinking and answers, reason over them carefully and provide a final integration result."
        )
        thinking2, answer2 = await final_decision_agent2(
            [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
            "Sub-task 2:" + final_instr2, is_sub_task=True
        )
        agents.append(f"Final Decision agent {final_decision_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
        subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
        logs.append(subtask_desc2)
        print("Step 2: ", sub_tasks[-1])

        # Sub-task 3: Sum branch contributions with Reflexion
        reflect_inst = (
            "Given previous attempts and feedback, carefully consider where you could go wrong in your latest sums. "
            "Using insights, compute Φ_total_700_800 and Φ_total_800_900 by summing branch integrals."
        )
        cot_reflect_instruction = "Sub-task 3: Sum branch contributions for each band." + reflect_inst
        cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                  model=self.node_model, temperature=0.0)
        critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent",
                                     model=self.node_model, temperature=0.0)
        subtask_desc3 = {
            "subtask_id": "subtask_3",
            "instruction": cot_reflect_instruction,
            "context": ["user query", "thinking2", "answer2"],
            "agent_collaboration": "Reflexion"
        }
        thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_reflect_instruction, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        for i in range(self.max_round):
            feedback, correct = await critic_agent3([taskInfo, thinking3, answer3],
                                        "Please review the answer above and criticize on where might be wrong. "
                                        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                        i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback.content}; correct: {correct.content}")
            if correct.content == "True":
                break
            thinking3, answer3 = await cot_agent3(
                [taskInfo, thinking2, answer2, thinking3, answer3, feedback],
                cot_reflect_instruction, i+1, is_sub_task=True
            )
            agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining: {thinking3.content}; answer: {answer3.content}")
        sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
        subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
        logs.append(subtask_desc3)
        print("Step 3: ", sub_tasks[-1])

        # Sub-task 4: Remove B-8 via SC-CoT and compute ratio
        sc_instruction = (
            "Sub-task 4: From baseline totals, subtract the B-8 contributions to get new fluxes Φ'_700_800 and Φ'_800_900, "
            "then compute R = Φ'_700_800 / Φ'_800_900. Ensure numeric precision."
        )
        cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                     model=self.node_model, temperature=0.5)
                       for _ in range(self.max_sc)]
        subtask_desc4 = {
            "subtask_id": "subtask_4",
            "instruction": sc_instruction,
            "context": ["user query", "thinking2", "answer2", "thinking3", "answer3"],
            "agent_collaboration": "SC_CoT"
        }
        answers4 = []
        thinkings4 = []
        for agent in cot_agents4:
            thinking4, answer4 = await agent([taskInfo, thinking2, answer2, thinking3, answer3],
                                              sc_instruction, is_sub_task=True)
            agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
            thinkings4.append(thinking4)
            answers4.append(answer4)
        final_decision4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                       model=self.node_model, temperature=0.0)
        thinking4, answer4 = await final_decision4(
            [taskInfo, thinking2, answer2, thinking3, answer3] + thinkings4 + answers4,
            "Sub-task 4: Synthesize and choose the most consistent answer for removal and ratio.",
            is_sub_task=True
        )
        agents.append(f"Final Decision agent {final_decision4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
        subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
        logs.append(subtask_desc4)
        print("Step 4: ", sub_tasks[-1])

        # Sub-task 5: Compare ratio to answer choices via Chain-of-Thought
        cot5_instruction = (
            "Sub-task 5: Compare the computed ratio R to choices {0.01, 0.1, 1, 10}. "
            "Provide a concise numeric justification for the nearest choice."
        )
        cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                  model=self.node_model, temperature=0.0)
        subtask_desc5 = {
            "subtask_id": "subtask_5",
            "instruction": cot5_instruction,
            "context": ["user query", "thinking4", "answer4"],
            "agent_collaboration": "CoT"
        }
        thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot5_instruction, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
        sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
        subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
        logs.append(subtask_desc5)
        print("Step 5: ", sub_tasks[-1])

        final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
        return final_answer, logs
