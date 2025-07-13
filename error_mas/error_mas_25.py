import asyncio
from collections import Counter

class LLMAgentBase:
    def __init__(self, outputs, role, model=None, temperature=0.0):
        self.outputs = outputs
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = f"agent_{id(self)}"

    async def __call__(self, inputs, instruction, round_idx=0, is_sub_task=False):
        # Dummy implementation for testing
        # In real use, this would call the LLM with inputs and instruction
        thinking = type('obj', (object,), {'content': f"Thinking about: {instruction[:50]}..."})
        answer = type('obj', (object,), {'content': f"Answer for: {instruction[:50]}..."})
        return thinking, answer

class Workflow:
    def __init__(self, node_model, debate_role, max_sc, max_round):
        self.node_model = node_model
        self.debate_role = debate_role
        self.max_sc = max_sc
        self.max_round = max_round

    async def make_final_answer(self, thinking, answer, sub_tasks, agents):
        return {
            "final_thinking": thinking.content,
            "final_answer": answer.content,
            "sub_tasks": sub_tasks,
            "agents": agents
        }

    async def forward_25(self, taskInfo):
        sub_tasks = []
        agents = []
        logs = []

        # Stage 1: Geometric Properties and Vector Relations

        # Sub-task 1: Debate - Identify and clearly state all given geometric properties
        debate_instr_1 = (
            "Sub-task 1: Identify and clearly state all given geometric properties of the hexagon ABCDEF, "
            "including convexity, equilateral side lengths, and the parallelism of opposite sides. "
            "Emphasize that no assumptions beyond those explicitly stated should be made, and clarify the implications of these properties on the polygon's shape and side directions. "
            "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        )
        debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
        N_max_1 = self.max_round
        all_thinking_1 = [[] for _ in range(N_max_1)]
        all_answer_1 = [[] for _ in range(N_max_1)]
        subtask_desc1 = {
            "subtask_id": "stage1_subtask1",
            "instruction": debate_instr_1,
            "context": ["user query"],
            "agent_collaboration": "Debate"
        }
        for r in range(N_max_1):
            for i, agent in enumerate(debate_agents_1):
                if r == 0:
                    thinking1, answer1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
                else:
                    input_infos_1 = [taskInfo] + all_thinking_1[r-1]
                    thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
                all_thinking_1[r].append(thinking1)
                all_answer_1[r].append(answer1)
        final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1], "Sub-task 1: Synthesize and choose the most consistent answer for geometric properties." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
        subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
        logs.append(subtask_desc1)
        print("Step 1: ", sub_tasks[-1])

        # Sub-task 2: SC_CoT - Express hexagon's side vectors and parallelism conditions
        cot_sc_instruction_2 = (
            "Sub-task 2: Express the hexagon's side vectors in vector form, defining vectors a = AB, b = BC, c = CD, d = DE, e = EF, f = FA. "
            "Formulate the true closure relation a + b + c + d + e + f = 0, and explicitly write the parallelism conditions for opposite sides (AB || DE, BC || EF, CD || FA) in vector terms. "
            "Avoid assuming any specific angle measures or vector sums without derivation."
        )
        N2 = self.max_sc
        cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N2)]
        possible_answers_2 = []
        possible_thinkings_2 = []
        subtask_desc2 = {
            "subtask_id": "stage1_subtask2",
            "instruction": cot_sc_instruction_2,
            "context": ["user query", thinking1.content],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N2):
            thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content], cot_sc_instruction_2, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
            possible_answers_2.append(answer2)
            possible_thinkings_2.append(thinking2)
        final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking2, answer2 = await final_decision_agent_2([taskInfo] + [t.content for t in possible_thinkings_2], "Sub-task 2: Synthesize and choose the most consistent vector expressions." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
        subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
        logs.append(subtask_desc2)
        print("Step 2: ", sub_tasks[-1])

        # New Sub-task 3: Debate - Explicitly verify and derive correct vector and angular relationships among AB, CD, EF
        debate_instr_3 = (
            "Sub-task 3: Using the true closure relation a + b + c + d + e + f = 0 and the parallelism conditions, explicitly verify and derive the correct vector and angular relationships among the lines AB, CD, and EF. "
            "Do not assume that AB + CD + EF = 0. Provide explicit vector formulas or angle measures between AB, CD, and EF. "
            "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        )
        debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
        N_max_3 = self.max_round
        all_thinking_3 = [[] for _ in range(N_max_3)]
        all_answer_3 = [[] for _ in range(N_max_3)]
        subtask_desc3 = {
            "subtask_id": "stage1_subtask3",
            "instruction": debate_instr_3,
            "context": ["user query", thinking2.content],
            "agent_collaboration": "Debate"
        }
        for r in range(N_max_3):
            for i, agent in enumerate(debate_agents_3):
                if r == 0:
                    thinking3, answer3 = await agent([taskInfo, thinking2.content], debate_instr_3, r, is_sub_task=True)
                else:
                    input_infos_3 = [taskInfo, thinking2.content] + all_thinking_3[r-1]
                    thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
                all_thinking_3[r].append(thinking3)
                all_answer_3[r].append(answer3)
        final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2.content] + all_thinking_3[-1], "Sub-task 3: Synthesize and confirm correct vector and angular relations." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
        subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
        logs.append(subtask_desc3)
        print("Step 3: ", sub_tasks[-1])

        # Sub-task 4: Debate - Define construction of the triangle formed by extended lines AB, CD, EF
        debate_instr_4 = (
            "Sub-task 4: Define the construction of the triangle formed by the intersections of the extended lines of sides AB, CD, and EF. "
            "Clarify how these lines intersect outside the hexagon and how the triangle's side lengths relate to the directions and positions of these lines. "
            "Avoid assumptions about the triangle's shape or side length relations without explicit derivation. "
            "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        )
        debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
        N_max_4 = self.max_round
        all_thinking_4 = [[] for _ in range(N_max_4)]
        all_answer_4 = [[] for _ in range(N_max_4)]
        subtask_desc4 = {
            "subtask_id": "stage1_subtask4",
            "instruction": debate_instr_4,
            "context": ["user query", thinking3.content],
            "agent_collaboration": "Debate"
        }
        for r in range(N_max_4):
            for i, agent in enumerate(debate_agents_4):
                if r == 0:
                    thinking4, answer4 = await agent([taskInfo, thinking3.content], debate_instr_4, r, is_sub_task=True)
                else:
                    input_infos_4 = [taskInfo, thinking3.content] + all_thinking_4[r-1]
                    thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
                all_thinking_4[r].append(thinking4)
                all_answer_4[r].append(answer4)
        final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3.content] + all_thinking_4[-1], "Sub-task 4: Synthesize and define triangle construction." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
        subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
        logs.append(subtask_desc4)
        print("Step 4: ", sub_tasks[-1])

        # Sub-task 5: SC_CoT - Assign coordinate system and parametric representation
        cot_sc_instruction_5 = (
            "Sub-task 5: Assign a coordinate system or parametric representation to the hexagon vertices and sides based on the vector relations derived earlier. "
            "Use this to compute the explicit parametric equations of the lines extending AB, CD, and EF. "
            "This setup should enable precise calculation of intersection points and distances in subsequent steps."
        )
        N5 = self.max_sc
        cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N5)]
        possible_answers_5 = []
        possible_thinkings_5 = []
        subtask_desc5 = {
            "subtask_id": "stage1_subtask5",
            "instruction": cot_sc_instruction_5,
            "context": ["user query", thinking4.content],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N5):
            thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4.content], cot_sc_instruction_5, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_5[i].id}, thinking: {thinking5.content}; answer: {answer5.content}")
            possible_answers_5.append(answer5)
            possible_thinkings_5.append(thinking5)
        final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking5, answer5 = await final_decision_agent_5([taskInfo] + [t.content for t in possible_thinkings_5], "Sub-task 5: Synthesize and assign coordinate system." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
        subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
        logs.append(subtask_desc5)
        print("Step 5: ", sub_tasks[-1])

        # Sub-task 6: SC_CoT - Calculate intersection points of extended lines
        cot_sc_instruction_6 = (
            "Sub-task 6: Calculate the intersection points of the extended lines AB, CD, and EF using the parametric equations. "
            "Derive explicit formulas for the coordinates of these intersection points as functions of the hexagon side length s and the vectors defined earlier. "
            "Ensure all calculations are symbolic or numeric as appropriate, avoiding unverified assumptions."
        )
        N6 = self.max_sc
        cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N6)]
        possible_answers_6 = []
        possible_thinkings_6 = []
        subtask_desc6 = {
            "subtask_id": "stage1_subtask6",
            "instruction": cot_sc_instruction_6,
            "context": ["user query", thinking5.content],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N6):
            thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5.content], cot_sc_instruction_6, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_6[i].id}, thinking: {thinking6.content}; answer: {answer6.content}")
            possible_answers_6.append(answer6)
            possible_thinkings_6.append(thinking6)
        final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking6, answer6 = await final_decision_agent_6([taskInfo] + [t.content for t in possible_thinkings_6], "Sub-task 6: Synthesize and calculate intersection points." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
        subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
        logs.append(subtask_desc6)
        print("Step 6: ", sub_tasks[-1])

        # Sub-task 7: SC_CoT - Compute distances between intersection points and extract constants k_i
        cot_sc_instruction_7 = (
            "Sub-task 7: Compute the distances between the intersection points obtained in the previous subtask to express the side lengths of the triangle formed by the extended lines. "
            "Derive explicit expressions for these distances as functions of the hexagon side length s and the previously defined vectors. "
            "Extract the constants k_i such that each triangle side length equals k_i * s."
        )
        N7 = self.max_sc
        cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N7)]
        possible_answers_7 = []
        possible_thinkings_7 = []
        subtask_desc7 = {
            "subtask_id": "stage1_subtask7",
            "instruction": cot_sc_instruction_7,
            "context": ["user query", thinking6.content],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N7):
            thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6.content], cot_sc_instruction_7, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_7[i].id}, thinking: {thinking7.content}; answer: {answer7.content}")
            possible_answers_7.append(answer7)
            possible_thinkings_7.append(thinking7)
        final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking7, answer7 = await final_decision_agent_7([taskInfo] + [t.content for t in possible_thinkings_7], "Sub-task 7: Synthesize and extract constants k_i." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
        subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
        logs.append(subtask_desc7)
        print("Step 7: ", sub_tasks[-1])

        # Sub-task 8: Reflexion - Critically evaluate and confirm vector relations and constants
        reflect_inst_8 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
        cot_reflect_instruction_8 = (
            "Sub-task 8: Critically evaluate and confirm the vector relations and constants derived so far. "
            "Verify that the vectors AB, CD, and EF do not sum to zero, and that the constants k_i correctly relate the hexagon side length s to the triangle side lengths 200, 240, and 300. "
            "This subtask should identify and correct any inconsistencies before proceeding to solve the system. "
            + reflect_inst_8
        )
        cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
        N_max_8 = self.max_round
        cot_inputs_8 = [taskInfo, thinking7.content, answer7.content]
        subtask_desc8 = {
            "subtask_id": "stage1_subtask8",
            "instruction": cot_reflect_instruction_8,
            "context": ["user query", thinking7.content, answer7.content],
            "agent_collaboration": "Reflexion"
        }
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, thinking: {thinking8.content}; answer: {answer8.content}")
        critic_inst_8 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        for i in range(N_max_8):
            feedback8, correct8 = await critic_agent_8([taskInfo, thinking8.content], critic_inst_8, i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent_8.id}, feedback: {feedback8.content}; correct: {correct8.content}")
            if correct8.content == "True":
                break
            cot_inputs_8.extend([thinking8.content, feedback8.content])
            thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
            agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining thinking: {thinking8.content}; answer: {answer8.content}")
        sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
        subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
        logs.append(subtask_desc8)
        print("Step 8: ", sub_tasks[-1])

        # Stage 2: Solve for side length s

        # Sub-task 1: CoT - Set up system of equations relating s to triangle side lengths
        cot_instruction_21 = (
            "Sub-task 1: Set up the system of equations relating the hexagon side length s to the known triangle side lengths 200, 240, and 300 using the constants k_i derived previously. "
            "Ensure the system correctly reflects the geometric constraints and vector relations without relying on prior incorrect assumptions."
        )
        cot_agent_21 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc21 = {
            "subtask_id": "stage2_subtask1",
            "instruction": cot_instruction_21,
            "context": ["user query", thinking8.content],
            "agent_collaboration": "CoT"
        }
        thinking21, answer21 = await cot_agent_21([taskInfo, thinking8.content], cot_instruction_21, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent_21.id}, thinking: {thinking21.content}; answer: {answer21.content}")
        sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking21.content}; answer - {answer21.content}")
        subtask_desc21['response'] = {"thinking": thinking21, "answer": answer21}
        logs.append(subtask_desc21)
        print("Step 9: ", sub_tasks[-1])

        # Sub-task 2: Debate - Solve system of equations for s
        debate_instr_22 = (
            "Sub-task 2: Solve the system of equations to find the numerical value of the hexagon's common side length s. "
            "Confirm that the solution is consistent with the convexity, equilateral conditions, and the geometric configuration of the hexagon and the triangle formed by the extended sides. "
            "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        )
        debate_agents_22 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
        N_max_22 = self.max_round
        all_thinking_22 = [[] for _ in range(N_max_22)]
        all_answer_22 = [[] for _ in range(N_max_22)]
        subtask_desc22 = {
            "subtask_id": "stage2_subtask2",
            "instruction": debate_instr_22,
            "context": ["user query", thinking21.content],
            "agent_collaboration": "Debate"
        }
        for r in range(N_max_22):
            for i, agent in enumerate(debate_agents_22):
                if r == 0:
                    thinking22, answer22 = await agent([taskInfo, thinking21.content], debate_instr_22, r, is_sub_task=True)
                else:
                    input_infos_22 = [taskInfo, thinking21.content] + all_thinking_22[r-1]
                    thinking22, answer22 = await agent(input_infos_22, debate_instr_22, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking22.content}; answer: {answer22.content}")
                all_thinking_22[r].append(thinking22)
                all_answer_22[r].append(answer22)
        final_decision_agent_22 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking22, answer22 = await final_decision_agent_22([taskInfo, thinking21.content] + all_thinking_22[-1], "Sub-task 2: Synthesize and solve for s." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking22.content}; answer - {answer22.content}")
        subtask_desc22['response'] = {"thinking": thinking22, "answer": answer22}
        logs.append(subtask_desc22)
        print("Step 10: ", sub_tasks[-1])

        # Sub-task 3: Reflexion - Validate computed side length s
        reflect_inst_23 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
        cot_reflect_instruction_23 = (
            "Sub-task 3: Validate the computed side length s by reconstructing the hexagon and the triangle formed by the extended sides using the derived value. "
            "Check that all geometric properties hold, including parallelism, convexity, and the given triangle side lengths, to confirm the correctness and consistency of the solution. "
            + reflect_inst_23
        )
        cot_agent_23 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        critic_agent_23 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
        N_max_23 = self.max_round
        cot_inputs_23 = [taskInfo, thinking22.content, answer22.content]
        subtask_desc23 = {
            "subtask_id": "stage2_subtask3",
            "instruction": cot_reflect_instruction_23,
            "context": ["user query", thinking22.content, answer22.content],
            "agent_collaboration": "Reflexion"
        }
        thinking23, answer23 = await cot_agent_23(cot_inputs_23, cot_reflect_instruction_23, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_23.id}, thinking: {thinking23.content}; answer: {answer23.content}")
        critic_inst_23 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        for i in range(N_max_23):
            feedback23, correct23 = await critic_agent_23([taskInfo, thinking23.content], critic_inst_23, i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent_23.id}, feedback: {feedback23.content}; correct: {correct23.content}")
            if correct23.content == "True":
                break
            cot_inputs_23.extend([thinking23.content, feedback23.content])
            thinking23, answer23 = await cot_agent_23(cot_inputs_23, cot_reflect_instruction_23, i + 1, is_sub_task=True)
            agents.append(f"Reflexion CoT agent {cot_agent_23.id}, refining thinking: {thinking23.content}; answer: {answer23.content}")
        sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking23.content}; answer - {answer23.content}")
        subtask_desc23['response'] = {"thinking": thinking23, "answer": answer23}
        logs.append(subtask_desc23)
        print("Step 11: ", sub_tasks[-1])

        final_answer = await self.make_final_answer(thinking23, answer23, sub_tasks, agents)
        return final_answer, logs

async def main():
    node_model = "dummy-model"
    debate_role = ["role1", "role2"]
    max_sc = 3
    max_round = 2
    workflow = Workflow(node_model, debate_role, max_sc, max_round)
    taskInfo = "Let ABCDEF be a convex equilateral hexagon in which all pairs of opposite sides are parallel. The triangle whose sides are extensions of segments AB, CD, and EF has side lengths 200, 240, and 300. Find the side length of the hexagon."
    final_answer, logs = await workflow.forward_25(taskInfo)
    print("\nFinal Answer:", final_answer)

if __name__ == '__main__':
    asyncio.run(main())